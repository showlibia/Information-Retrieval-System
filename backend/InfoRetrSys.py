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
        self.inverted_index = defaultdict(lambda: defaultdict(list))  # term -> {doc_id: [positions]}
        self.doc_texts = {}  # doc_id -> original text
        self.doc_freq = defaultdict(int)  # term -> number of documents containing term
        self.total_docs = 0
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.current_doc_name = None
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)

    def process_pdf(self, pdf_path):
        self.current_doc_name = os.path.basename(pdf_path)  # Store the filename
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
            for pos, token in enumerate(tokens):
                self.inverted_index[token][doc_id].append(pos)
                if pos == 0 or token not in tokens[:pos]:
                    self.doc_freq[token] += 1

    def format_chunk(self, doc_id, score=None):
        """Create a standardized chunk dictionary with metadata."""
        page_num = int(doc_id.split("_")[1])  # Extract page number from doc_id
        metadata = {
            "doc": self.current_doc_name,  # Document name
            "page": page_num              # Page number
        }
        chunk = {
            "doc_id": doc_id,
            "text": self.doc_texts[doc_id][:100] + "...",
            "metadata": metadata
        }
        if score is not None:
            chunk["score"] = score
        return chunk

    def global_search(self, query_tokens):
        doc_scores = defaultdict(float)
        for term in query_tokens:
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    doc_scores[doc_id] += self.compute_tf_idf(term, doc_id, query_tokens)
        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.format_chunk(doc_id, score) for doc_id, score in results]
    
    def top_k_search(self, query_tokens, k):
        doc_scores = defaultdict(float)
        for term in query_tokens:
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    doc_scores[doc_id] += self.compute_tf_idf(term, doc_id, query_tokens)
        heap = []
        for doc_id, score in doc_scores.items():
            heapq.heappush(heap, (score, doc_id))
            if len(heap) > k:
                heapq.heappop(heap)
        results = sorted(heap, reverse=True)
        return [self.format_chunk(doc_id, score) for score, doc_id in results]

    def phrase_query(self, query_tokens):
        results = []
        if query_tokens[0] not in self.inverted_index:
            return results
        candidate_docs = set(self.inverted_index[query_tokens[0]].keys())
        for doc_id in candidate_docs:
            positions = [self.inverted_index[term][doc_id] for term in query_tokens 
                         if doc_id in self.inverted_index[term]]
            if len(positions) == len(query_tokens):
                for start_pos in positions[0]:
                    match = True
                    for i, term_positions in enumerate(positions[1:], 1):
                        if start_pos + i not in term_positions:
                            match = False
                            break
                    if match:
                        results.append(doc_id)
                        break
        return [self.format_chunk(doc_id) for doc_id in results]

    def wildcard_query(self, query):
        pattern = query.lower().replace("*", ".*")
        results = defaultdict(float)
        for term in self.inverted_index:
            if re.match(pattern, term):
                for doc_id in self.inverted_index[term]:
                    results[doc_id] += self.compute_tf_idf(term, doc_id, [term])
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [self.format_chunk(doc_id, score) for doc_id, score in sorted_results]

    def boolean_query(self, query):
        parts = query.lower().split()
        if "and" in parts:
            terms = [self.stemmer.stem(t) for t in parts if t not in ["and"]]
            doc_sets = [set(self.inverted_index[term].keys()) for term in terms if term in self.inverted_index]
            if not doc_sets:
                return []
            common_docs = set.intersection(*doc_sets)
            return [self.format_chunk(doc_id) for doc_id in common_docs]
        elif "or" in parts:
            terms = [self.stemmer.stem(t) for t in parts if t not in ["or"]]
            all_docs = set()
            for term in terms:
                if term in self.inverted_index:
                    all_docs.update(self.inverted_index[term].keys())
            return [self.format_chunk(doc_id) for doc_id in all_docs]
        else:
            query_tokens = [self.stemmer.stem(query)]
            return self.global_search(query_tokens)

    def retrieve(self, query, query_type="global", top_k=3):
        query_tokens = self.preprocess_text("query", query)
        if query_type == "global":
            return self.global_search(query_tokens)
        elif query_type == "top_k":
            return self.top_k_search(query_tokens, top_k)
        elif query_type == "phrase":
            return self.phrase_query(query_tokens)
        elif query_type == "wildcard":
            return self.wildcard_query(query)
        elif query_type == "boolean":
            return self.boolean_query(query)
        else:
            return {"error": "Unsupported query type"}

    def compute_tf_idf(self, term, doc_id, query_terms):
        tf = len(self.inverted_index[term][doc_id]) if doc_id in self.inverted_index[term] else 0
        df = self.doc_freq[term]
        idf = math.log(self.total_docs / (df + 1)) if df > 0 else 0
        return tf * idf
    
    def select_query_type(self, query):
        """Use GPT to select the most appropriate query type for the given user query."""
        prompt = (
            f"User Query: '{query}'. Please select the most appropriate query method: "
            "Global Search, TOP K Search, Phrase Query, Wildcard Query, Synonym Query, Boolean Query."
            "Only return the name of the selected method."
        )
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            selected_type = response.choices[0].message.content.strip().lower()
            valid_types = ["global", "top_k", "phrase", "wildcard", "synonym", "boolean"]
            return selected_type if selected_type in valid_types else "global"
        except Exception as e:
            print(f"Error selecting query type: {e}")
            return "global"  # Default to global search on error

    def synonym_query(self, query_tokens):
        expanded_terms = set(query_tokens)
        for token in query_tokens:
            synsets = wordnet.synsets(token)
            for syn in synsets:
                for lemma in syn.lemmas():
                    expanded_terms.add(self.stemmer.stem(lemma.name()))
        return self.global_search(list(expanded_terms))

    def generate_answer(self, query, chunks):
        """Generate an answer using ChatGPT based on retrieved chunks."""
        context = "\n".join([chunk["text"] for chunk in chunks])
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