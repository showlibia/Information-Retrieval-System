# backend/InfoRetrSys.py
import re
import os
from collections import defaultdict
import math
import heapq
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from openai import OpenAI

# Ensure you have NLTK corpora downloaded:
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

base_url = 'https://ark.cn-beijing.volces.com/api/v3/'
MODEL_NAME = 'deepseek-v3-250324'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

class InfoRetrievalSystem:
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(list))  # term -> {doc_id: [positions]}
        self.doc_texts = {}  # doc_id -> original text (now stores sentences)
        self.doc_metadata = {} # doc_id -> {"original_doc": "filename", "page_num": pagenum, "sentence_idx_in_page": idx}
        self.doc_freq = defaultdict(int)  # term -> number of documents containing term
        self.total_docs = 0
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.current_doc_name = None
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)

        # DEBUG: Check if 'model' or 'name' are in stopwords
        print(f"DEBUG: 'model' in stopwords: {'model' in self.stop_words}")
        print(f"DEBUG: 'name' in stopwords: {'name' in self.stop_words}")


    def process_pdf(self, pdf_path):
        self.current_doc_name = os.path.basename(pdf_path)
        print(f"Processing {pdf_path}...")
        self.clear_index() # Clear existing index before processing new PDF
        sentences = self.parse_pdf_to_sentences(pdf_path)
        self.build_index(sentences)
        print("Index built successfully.")
        print(f"DEBUG: Total documents (sentences) after processing: {self.total_docs}")
        print(f"DEBUG: Sample doc_texts keys (first 5): {list(self.doc_texts.keys())[:5]}")
        print(f"DEBUG: Sample inverted_index entries (first 5 terms):")
        for i, (term, docs) in enumerate(self.inverted_index.items()):
            if i >= 5: break
            print(f"  Term '{term}': {len(docs)} documents")
        print(f"DEBUG: Sample doc_freq entries (first 5 terms):")
        for i, (term, freq) in enumerate(self.doc_freq.items()):
            if i >= 5: break
            print(f"  Term '{term}': {freq} documents")


    def clear_index(self):
        self.inverted_index.clear()
        self.doc_texts.clear()
        self.doc_metadata.clear()
        self.doc_freq.clear()
        self.total_docs = 0
        print("DEBUG: Index cleared.")

    def parse_pdf_to_sentences(self, pdf_path):
        reader = PdfReader(pdf_path)
        all_sentences = []
        sentence_counter = 0
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                sentences_on_page = sent_tokenize(page_text)
                for i, sentence in enumerate(sentences_on_page):
                    doc_id = f"sentence_{sentence_counter}"
                    self.doc_texts[doc_id] = sentence
                    self.doc_metadata[doc_id] = {
                        "original_doc": self.current_doc_name,
                        "page_num": page_num,
                        "sentence_idx_in_page": i
                    }
                    all_sentences.append((doc_id, sentence))
                    sentence_counter += 1
        print(f"DEBUG: Parsed {len(all_sentences)} sentences from PDF.")
        return all_sentences

    def preprocess_text(self, doc_id, text):
        text = re.sub(r'\s+', ' ', text).strip().lower()
        tokens = word_tokenize(text)
        
        # DEBUG: Check tokens before filtering
        # print(f"DEBUG preprocess_text: raw tokens for {doc_id}: {tokens[:10]}...")

        tokens = [self.stemmer.stem(token) for token in tokens
                  if token.isalnum() and token not in self.stop_words]
        
        # DEBUG: Check tokens after filtering
        # print(f"DEBUG preprocess_text: filtered tokens for {doc_id}: {tokens[:10]}...")
        return tokens

    def build_index(self, documents):
        self.total_docs = len(documents)
        for doc_id, text in documents:
            tokens = self.preprocess_text(doc_id, text)
            # DEBUG: Print tokens for a few documents to see if they are empty
            # if self.total_docs > 0 and (list(self.doc_texts.keys()).index(doc_id) < 5 or list(self.doc_texts.keys()).index(doc_id) > self.total_docs - 5):
            #     print(f"DEBUG build_index: tokens for doc '{doc_id}': {tokens[:10]}...")

            for pos, token in enumerate(tokens):
                self.inverted_index[token][doc_id].append(pos)
                # Correct doc_freq update: count unique term per document
                if doc_id not in self.inverted_index[token]: # This ensures doc_freq is counted only once per doc
                    self.doc_freq[token] += 1
        
        # Final check on doc_freq
        # for term, freq in self.doc_freq.items():
        #     if freq == 0:
        #         print(f"WARNING: Term '{term}' has zero doc_freq.")


    def format_chunk(self, doc_id, score=None, query_method=None):
        """Create a standardized chunk dictionary with metadata."""
        metadata = self.doc_metadata.get(doc_id, {})
        chunk = {
            "doc_id": doc_id,
            "text": self.doc_texts[doc_id],
            "metadata": {
                "doc": metadata.get("original_doc", self.current_doc_name),
                "page": metadata.get("page_num", "unknown"),
                "sentence_index": metadata.get("sentence_idx_in_page", "unknown")
            }
        }
        if score is not None:
            chunk["score"] = score
        if query_method is not None:
            chunk["query_method"] = query_method
        return chunk

    def compute_tf_idf(self, term, doc_id):
        tf = len(self.inverted_index[term][doc_id]) if doc_id in self.inverted_index[term] else 0
        df = self.doc_freq[term]
        
        # DEBUG: Check values before IDF calculation
        # if tf > 0:
        #     print(f"DEBUG compute_tf_idf: term='{term}', doc_id='{doc_id}', tf={tf}, df={df}, total_docs={self.total_docs}")

        idf = math.log(self.total_docs / (df + 1)) if df > 0 and self.total_docs > 0 else 0
        
        # DEBUG: Check final TF-IDF score
        # if tf > 0 and idf > 0:
        #     print(f"DEBUG compute_tf_idf: score={tf * idf}")

        return tf * idf

    # --- Retrieval Methods with refined scoring ---

    def _calculate_base_tfidf_score(self, doc_id, terms_to_score):
        """Helper to calculate TF-IDF sum for a set of terms in a document."""
        score = 0.0
        for term in terms_to_score:
            score += self.compute_tf_idf(term, doc_id)
        return score

    def global_search(self, query_tokens):
        retrieved_chunks = []
        # DEBUG: Check query_tokens for global_search
        print(f"DEBUG global_search: query_tokens={query_tokens}")
        
        for doc_id in self.doc_texts.keys():
            score = self._calculate_base_tfidf_score(doc_id, query_tokens)
            if score > 0:
                chunk = self.format_chunk(doc_id, score, "global")
                retrieved_chunks.append(chunk)
        return retrieved_chunks

    def top_k_search(self, query_tokens, k=20):
        doc_scores = defaultdict(float)
        # DEBUG: Check query_tokens for top_k_search
        print(f"DEBUG top_k_search: query_tokens={query_tokens}")

        for doc_id in self.doc_texts.keys():
            score = self._calculate_base_tfidf_score(doc_id, query_tokens)
            if score > 0:
                doc_scores[doc_id] = score

        heap = []
        for doc_id, score in doc_scores.items():
            if score > 0:
                heapq.heappush(heap, (score, doc_id))
                if len(heap) > k:
                    heapq.heappop(heap)
        
        retrieved_chunks = []
        for score, doc_id in heap:
            chunk = self.format_chunk(doc_id, score, "top_k")
            retrieved_chunks.append(chunk)
        return retrieved_chunks


    def phrase_query(self, query_raw_tokens):
        results = []
        if not query_raw_tokens:
            print("DEBUG phrase_query: query_raw_tokens is empty.")
            return []

        stemmed_query_tokens = [self.stemmer.stem(token.lower()) for token in query_raw_tokens if token.isalnum()]
        print(f"DEBUG phrase_query: stemmed_query_tokens={stemmed_query_tokens}")

        if not stemmed_query_tokens:
            print("DEBUG phrase_query: stemmed_query_tokens is empty after processing.")
            return []

        if any(term not in self.inverted_index for term in stemmed_query_tokens):
            print(f"DEBUG phrase_query: Not all stemmed terms found in inverted index for phrase query: {stemmed_query_tokens}")
            return []

        candidate_docs = set(self.inverted_index[stemmed_query_tokens[0]].keys())
        # print(f"DEBUG phrase_query: Candidate docs for first term '{stemmed_query_tokens[0]}': {candidate_docs}")

        for doc_id in candidate_docs:
            positions = [self.inverted_index[term][doc_id] for term in stemmed_query_tokens]
            
            first_term_positions = positions[0]
            for start_pos in first_term_positions:
                match = True
                for i, term_positions in enumerate(positions[1:], 1):
                    if (start_pos + i) not in term_positions:
                        match = False
                        break
                if match:
                    phrase_score = self._calculate_base_tfidf_score(doc_id, stemmed_query_tokens)
                    phrase_score += 7.0
                    chunk = self.format_chunk(doc_id, phrase_score, "phrase")
                    results.append(chunk)
                    break
        return results

    def wildcard_query(self, query_string):
        pattern_str = query_string.lower().replace("*", ".*")
        print(f"DEBUG wildcard_query: pattern_str='{pattern_str}'")
        try:
            pattern = re.compile(pattern_str)
        except re.error:
            print(f"DEBUG wildcard_query: Invalid regex pattern: {pattern_str}")
            return []

        retrieved_chunks = []
        doc_scores = defaultdict(float)

        matched_terms = set()
        for term_in_index in self.inverted_index:
            if pattern.fullmatch(term_in_index):
                matched_terms.add(term_in_index)
        print(f"DEBUG wildcard_query: Matched terms from index: {list(matched_terms)[:5]}...")

        for term in matched_terms:
            for doc_id in self.inverted_index[term]:
                doc_scores[doc_id] += self.compute_tf_idf(term, doc_id)

        for doc_id, score in doc_scores.items():
            if score > 0:
                final_score = score + 2.0
                chunk = self.format_chunk(doc_id, final_score, "wildcard")
                retrieved_chunks.append(chunk)
        return retrieved_chunks


    def boolean_query(self, query_string):
        parts = query_string.lower().split()
        retrieved_chunks = []

        has_and = "and" in parts
        has_or = "or" in parts
        has_not = "not" in parts # Not implemented, but good to identify

        terms_raw = [t for t in parts if t not in ["and", "or", "not"]]
        stemmed_terms = [self.stemmer.stem(t) for t in terms_raw if t.isalnum()]
        print(f"DEBUG boolean_query: stemmed_terms={stemmed_terms}, has_and={has_and}, has_or={has_or}")

        if not stemmed_terms:
            print("DEBUG boolean_query: stemmed_terms is empty.")
            return []

        if has_and:
            doc_sets = [set(self.inverted_index[term].keys()) for term in stemmed_terms if term in self.inverted_index]
            if not doc_sets:
                print(f"DEBUG boolean_query (AND): One or more terms not found in index: {stemmed_terms}")
                return []
            common_docs = set.intersection(*doc_sets)
            print(f"DEBUG boolean_query (AND): Found {len(common_docs)} common documents.")

            for doc_id in common_docs:
                score = self._calculate_base_tfidf_score(doc_id, stemmed_terms)
                score += 6.0
                chunk = self.format_chunk(doc_id, score, "boolean")
                retrieved_chunks.append(chunk)

        elif has_or:
            all_docs = set()
            for term in stemmed_terms:
                if term in self.inverted_index:
                    all_docs.update(self.inverted_index[term].keys())
            print(f"DEBUG boolean_query (OR): Found {len(all_docs)} combined documents.")

            for doc_id in all_docs:
                score = self._calculate_base_tfidf_score(doc_id, stemmed_terms)
                score += 3.0
                chunk = self.format_chunk(doc_id, score, "boolean")
                retrieved_chunks.append(chunk)
        else:
            print("DEBUG boolean_query: No boolean operators, falling back to global search.")
            retrieved_chunks.extend(self.global_search(stemmed_terms))
        return retrieved_chunks

    def synonym_query(self, query_raw_tokens):
        expanded_terms = set()
        stemmed_query_tokens_original = [self.stemmer.stem(token.lower()) for token in query_raw_tokens if token.isalnum()]
        expanded_terms.update(stemmed_query_tokens_original)

        for token in query_raw_tokens:
            synsets = wordnet.synsets(token)
            for syn in synsets:
                for lemma in syn.lemmas():
                    stemmed_lemma = self.stemmer.stem(lemma.name().lower())
                    if stemmed_lemma.isalnum():
                        expanded_terms.add(stemmed_lemma)
        
        final_expanded_terms = list(expanded_terms)
        print(f"DEBUG synonym_query: Final expanded terms: {final_expanded_terms[:10]}...")

        retrieved_chunks = []
        doc_scores = defaultdict(float)

        for doc_id in self.doc_texts.keys():
            score = self._calculate_base_tfidf_score(doc_id, final_expanded_terms)
            if score > 0:
                doc_scores[doc_id] = score

        for doc_id, score in doc_scores.items():
            if score > 0: # Ensure we only add chunks that actually scored something
                final_score = score + 1.0
                chunk = self.format_chunk(doc_id, final_score, "synonym")
                retrieved_chunks.append(chunk)
        return retrieved_chunks


    def retrieve_and_score_all(self, query_string, top_n=5):
        print(f"DEBUG retrieve_and_score_all: Incoming query_string='{query_string}'")
        query_tokens = self.preprocess_text("query", query_string) # Stemmed tokens from query_string
        query_raw_tokens = word_tokenize(query_string) # Raw tokens for phrase/synonym

        # Check if preprocessing yields anything
        if not query_tokens and not query_raw_tokens:
            print("DEBUG retrieve_and_score_all: Query tokens are empty after preprocessing. Returning empty list.")
            return []

        all_results = {}

        # Run all retrieval methods and collect results
        print(f"Running global search for query: {query_string}")
        results_global = self.global_search(query_tokens)
        self.add_results_to_all(all_results, results_global)
        print(f"Global search added {len(results_global)} chunks. Current unique results: {len(all_results)}")

        print(f"Running top_k search for query: {query_string}")
        results_top_k = self.top_k_search(query_tokens, k=20)
        self.add_results_to_all(all_results, results_top_k)
        print(f"Top-K search added {len(results_top_k)} chunks. Current unique results: {len(all_results)}")

        print(f"Running phrase query for query: {query_string}")
        if len(query_raw_tokens) > 1:
             results_phrase = self.phrase_query(query_raw_tokens)
             self.add_results_to_all(all_results, results_phrase)
             print(f"Phrase query added {len(results_phrase)} chunks. Current unique results: {len(all_results)}")
        else:
             print("Skipping phrase query (single token).")


        print(f"Running wildcard query for query: {query_string}")
        if "*" in query_string:
            results_wildcard = self.wildcard_query(query_string)
            self.add_results_to_all(all_results, results_wildcard)
            print(f"Wildcard query added {len(results_wildcard)} chunks. Current unique results: {len(all_results)}")
        else:
            print("Skipping wildcard query (no '*' found).")

        print(f"Running boolean query for query: {query_string}")
        if "and" in query_string.lower() or "or" in query_string.lower() or "not" in query_string.lower():
            results_boolean = self.boolean_query(query_string)
            self.add_results_to_all(all_results, results_boolean)
            print(f"Boolean query added {len(results_boolean)} chunks. Current unique results: {len(all_results)}")
        else:
            print("Skipping boolean query (no boolean operators).")

        print(f"Running synonym query for query: {query_string}")
        results_synonym = self.synonym_query(query_raw_tokens)
        self.add_results_to_all(all_results, results_synonym)
        print(f"Synonym query added {len(results_synonym)} chunks. Current unique results: {len(all_results)}")

        # Sort all unique chunks by score in descending order
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        print(f"DEBUG retrieve_and_score_all: Total unique chunks before slicing: {len(sorted_results)}")
        print(f"DEBUG retrieve_and_score_all: Top {top_n} results scores: {[chunk['score'] for chunk in sorted_results[:top_n]]}")


        return sorted_results[:top_n]

    def add_results_to_all(self, all_results_dict, new_results_list):
        for chunk in new_results_list:
            doc_id = chunk["doc_id"]
            score = chunk.get("score", 0.0)
            method = chunk.get("query_method", "unknown")

            if doc_id not in all_results_dict or all_results_dict[doc_id]["score"] < score:
                all_results_dict[doc_id] = {
                    "doc_id": doc_id,
                    "text": self.doc_texts[doc_id],
                    "metadata": self.doc_metadata[doc_id],
                    "score": score,
                    "query_method": method
                }

    def generate_answer(self, query, chunks):
        context = "\n".join([f"({chunk.get('query_method', 'N/A')} Score: {chunk.get('score', 0.0):.2f}) {chunk['text']}" for chunk in chunks])
        if not context:
            return "No relevant context found to generate an answer."

        prompt = f"Based on the following context, answer the user's question concisely and directly. If the information is not in the context, state that you cannot answer from the provided information.\n\nUser Question: '{query}'\n\nContext:\n{context}\n\nAnswer:"
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Unable to generate an answer due to an error."