import re
import os
from collections import defaultdict
import math
import heapq
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from openai import OpenAI

base_url = 'https://ark.cn-beijing.volces.com/api/v3/'
MODEL_NAME = 'deepseek-v3-250324'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

class InfoRetrievalSystem:
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.doc_texts = {}
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.current_doc_name = None
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)

    def process_pdf(self, pdf_path):
        self.current_doc_name = os.path.basename(pdf_path)
        print(f"Processing {pdf_path}...")
        documents = self.parse_pdf(pdf_path)
        self.build_index(documents)
        print("Index built successfully.")

    def parse_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text_with_docs = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                doc_id = f"page_{page_num}"
                text_with_docs.append((doc_id, page_text))
        return text_with_docs

    def preprocess_text(self, doc_id, text):
        text = re.sub(r'\s+', ' ', text).strip().lower()
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens
                  if token.isalnum() and token not in self.stop_words]
        return tokens

    def build_index(self, documents):
        self.total_docs = len(documents)
        for doc_id, text in documents:
            self.doc_texts[doc_id] = text
            tokens = self.preprocess_text(doc_id, text)
            unique_terms_in_doc = set()
            for pos, token in enumerate(tokens):
                self.inverted_index[token][doc_id].append(pos)
                if token not in unique_terms_in_doc:
                    self.doc_freq[token] += 1
                    unique_terms_in_doc.add(token)

    # Format chunk for standardized output
    def format_chunk(self, doc_id, score=None, query_method="Global Search"):
        """Create a standardized chunk dictionary with metadata."""
        page_num = int(doc_id.split("_")[1])
        metadata = {
            "doc": self.current_doc_name,
            "page": page_num
        }
        chunk = {
            "doc_id": doc_id,
            "text": self.doc_texts[doc_id],
            "metadata": metadata,
            "query_methods": [query_method]
        }
        if score is not None:
            chunk["score"] = score
        return chunk

    def compute_tf_idf(self, term, doc_id):
        term_count_in_doc = len(self.inverted_index[term][doc_id]) if doc_id in self.inverted_index[term] else 0
        total_terms_in_doc = len(self.preprocess_text(doc_id, self.doc_texts[doc_id]))
        tf = term_count_in_doc / total_terms_in_doc if total_terms_in_doc > 0 else 0
        df = self.doc_freq[term]
        idf = math.log(self.total_docs / (df + 1)) if df > 0 else 0
        return tf * idf

    def compute_wf_idf(self, term, doc_id):
        term_count_in_doc = len(self.inverted_index[term][doc_id]) if doc_id in self.inverted_index[term] else 0
        wf = 1 + math.log(term_count_in_doc) if term_count_in_doc > 0 else 0
        df = self.doc_freq[term]
        idf = math.log(self.total_docs / (df + 1)) if df > 0 else 0
        return wf * idf

    # Compute relevance score for a chunk
    def compute_score_for_chunk(self, chunk, query_tokens, scoring_method="tf_idf"):
        """Compute the relevance score for a chunk based on query tokens. scoring_method: 'tf_idf', 'wf_idf', or 'combined'"""
        score = 0.0
        doc_id = chunk["doc_id"]
        unique_query_tokens = set(query_tokens)
        for term in unique_query_tokens:
            if scoring_method == "tf_idf":
                score += self.compute_tf_idf(term, doc_id)
            elif scoring_method == "wf_idf":
                score += self.compute_wf_idf(term, doc_id)
            elif scoring_method == "combined":
                tf_idf_score = self.compute_tf_idf(term, doc_id)
                wf_idf_score = self.compute_wf_idf(term, doc_id)
                score += (tf_idf_score + wf_idf_score) / 2
            else:
                score += self.compute_tf_idf(term, doc_id)
        return score * 1000

    def global_search(self, query_tokens):
        candidate_docs = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
        return [self.format_chunk(doc_id, query_method="Global Search") for doc_id in candidate_docs]

    def top_k_search(self, query_tokens, k):
        candidate_docs = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
        return [self.format_chunk(doc_id, query_method="TOP K Search") for doc_id in candidate_docs]

    def phrase_query(self, query_tokens):
        results = []
        if not query_tokens:
            return []
        candidate_docs = set(self.inverted_index[query_tokens[0]].keys())
        for term in query_tokens[1:]:
            if term in self.inverted_index:
                candidate_docs.intersection_update(self.inverted_index[term].keys())
            else:
                return []
        for doc_id in candidate_docs:
            term_positions_in_doc = []
            all_terms_present = True
            for term in query_tokens:
                if doc_id in self.inverted_index[term]:
                    term_positions_in_doc.append(self.inverted_index[term][doc_id])
                else:
                    all_terms_present = False
                    break
            if not all_terms_present:
                continue
            first_term_positions = term_positions_in_doc[0]
            for start_pos in first_term_positions:
                match = True
                for i in range(1, len(query_tokens)):
                    if (start_pos + i) not in term_positions_in_doc[i]:
                        match = False
                        break
                if match:
                    results.append(self.format_chunk(doc_id, query_method="Phrase Query"))
                    break
        return results

    def wildcard_query(self, query):
        pattern = query.lower().replace("*", ".*")
        results = set()
        for term in self.inverted_index:
            if re.match(pattern, term):
                for doc_id in self.inverted_index[term]:
                    results.add(doc_id)
        return [self.format_chunk(doc_id, query_method="Wildcard Query") for doc_id in results]

    def boolean_query(self, query):
        parts = query.lower().split()
        stemmed_parts = [self.stemmer.stem(p) for p in parts if p not in ["and", "or", "not"]]
        result_doc_ids = set()
        if "and" in parts:
            terms = [self.stemmer.stem(t) for t in parts if t not in ["and", "or", "not"]]
            doc_sets = [set(self.inverted_index[term].keys()) for term in terms if term in self.inverted_index]
            if doc_sets:
                result_doc_ids.update(set.intersection(*doc_sets))
        elif "or" in parts:
            terms = [self.stemmer.stem(t) for t in parts if t not in ["and", "or", "not"]]
            for term in terms:
                if term in self.inverted_index:
                    result_doc_ids.update(self.inverted_index[term].keys())
        elif "not" in parts:
            if len(parts) == 2 and parts[0] == "not":
                term_to_exclude = self.stemmer.stem(parts[1])
                excluded_docs = set(self.inverted_index[term_to_exclude].keys()) if term_to_exclude in self.inverted_index else set()
                all_possible_docs = set(self.doc_texts.keys())
                result_doc_ids.update(all_possible_docs - excluded_docs)
            else:
                positive_terms = [self.stemmer.stem(p) for p in parts if p not in ["and", "or", "not"] and parts[parts.index(p) - 1] != "not"]
                negative_terms = [self.stemmer.stem(parts[i+1]) for i, p in enumerate(parts) if p == "not"]
                temp_docs = set(self.doc_texts.keys())
                if positive_terms:
                    positive_doc_sets = [set(self.inverted_index[term].keys()) for term in positive_terms if term in self.inverted_index]
                    if positive_doc_sets:
                        temp_docs = set.intersection(*positive_doc_sets)
                    else:
                        temp_docs = set()
                for term in negative_terms:
                    if term in self.inverted_index:
                        docs_with_negative_term = set(self.inverted_index[term].keys())
                        temp_docs = temp_docs - docs_with_negative_term
                result_doc_ids.update(temp_docs)
        else:
            pass
        return [self.format_chunk(doc_id, query_method="Boolean Query") for doc_id in result_doc_ids]

    def synonym_query(self, query_tokens):
        expanded_terms = set(query_tokens)
        for token in query_tokens:
            synsets = wordnet.synsets(token)
            for syn in synsets:
                for lemma in syn.lemmas():
                    stemmed_lemma = self.stemmer.stem(lemma.name())
                    if stemmed_lemma.isalnum() and stemmed_lemma not in self.stop_words:
                        expanded_terms.add(stemmed_lemma)
        results = set()
        for term in expanded_terms:
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    results.add(doc_id)
        return [self.format_chunk(doc_id, query_method="Synonym Query") for doc_id in results]

    # Retrieve and score chunks using multiple query methods
    def retrieve_and_score_all(self, query, top_n=5, scoring_method="tf_idf"):
        query_tokens = self.preprocess_text("query", query)
        all_raw_chunks_by_method = defaultdict(list)
        all_raw_chunks_by_method["Global Search"] = self.global_search(query_tokens)
        all_raw_chunks_by_method["Phrase Query"] = self.phrase_query(query_tokens)
        all_raw_chunks_by_method["Wildcard Query"] = self.wildcard_query(query)
        all_raw_chunks_by_method["Boolean Query"] = self.boolean_query(query)
        all_raw_chunks_by_method["Synonym Query"] = self.synonym_query(query_tokens)
        consolidated_chunks_map = defaultdict(lambda: {
            "doc_id": None,
            "text": None,
            "metadata": {},
            "query_methods": set(),
            "score": 0.0
        })
        for method, chunks_list in all_raw_chunks_by_method.items():
            for chunk in chunks_list:
                doc_id = chunk['doc_id']
                if consolidated_chunks_map[doc_id]["doc_id"] is None:
                    consolidated_chunks_map[doc_id]["doc_id"] = doc_id
                    consolidated_chunks_map[doc_id]["text"] = chunk['text']
                    consolidated_chunks_map[doc_id]["metadata"] = chunk['metadata']
                consolidated_chunks_map[doc_id]["query_methods"].add(method)
        scored_final_results = []
        for doc_id, chunk_data in consolidated_chunks_map.items():
            chunk_data['query_methods'] = sorted(list(chunk_data['query_methods']))
            chunk_score = self.compute_score_for_chunk(chunk_data, query_tokens, scoring_method)
            chunk_data['score'] = chunk_score
            scored_final_results.append(chunk_data)
        sorted_results = sorted(scored_final_results, key=lambda x: x['score'], reverse=True)
        top_n_overall_chunks = sorted_results[:top_n]
        top_n_doc_ids = {chunk['doc_id'] for chunk in top_n_overall_chunks}
        other_method_chunks = []
        all_processed_doc_ids = set()
        for method, chunks_list in all_raw_chunks_by_method.items():
            method_specific_chunks = []
            for chunk in chunks_list:
                if chunk['doc_id'] not in top_n_doc_ids and chunk['doc_id'] not in all_processed_doc_ids:
                    original_consolidated_chunk = consolidated_chunks_map.get(chunk['doc_id'])
                    if original_consolidated_chunk:
                        chunk['score'] = original_consolidated_chunk['score']
                        chunk['query_methods'] = original_consolidated_chunk['query_methods']
                    else:
                        chunk['score'] = self.compute_score_for_chunk(chunk, query_tokens, scoring_method)
                    method_specific_chunks.append(chunk)
                    all_processed_doc_ids.add(chunk['doc_id'])
                if len(method_specific_chunks) >= 3:
                    break
            other_method_chunks.extend(method_specific_chunks)
        for chunk in top_n_overall_chunks:
            chunk['text_preview'] = chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text']
        for chunk in other_method_chunks:
            chunk['text_preview'] = chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text']
        return top_n_overall_chunks, other_method_chunks

    # Generate answer using LLM
    def generate_answer(self, query, chunks):
        """Generate an answer using ChatGPT based on retrieved chunks."""
        context = "\n".join([chunk["text_preview"] for chunk in chunks])
        prompt = f"Answer the user's question based on the following context: '{query}'\n\nContext:\n{context}"
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Unable to generate an answer due to an error."