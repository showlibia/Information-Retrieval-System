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


    def process_pdf(self, pdf_path):
        self.current_doc_name = os.path.basename(pdf_path)
        print(f"Processing {pdf_path}...")
        self.clear_index() # Clear existing index before processing new PDF
        sentences_data = self.parse_pdf_to_sentences(pdf_path)
        self.build_index(sentences_data)
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
        all_sentences_data = [] # Stores (doc_id, sentence_text)
        sentence_counter = 0 # Global sentence counter for unique doc_ids

        for page_num, page in enumerate(reader.pages, 1):
            page_text_extracted = page.extract_text()
            if not page_text_extracted:
                continue

            page_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', page_text_extracted)
            
            page_text = re.sub(r'[ \t]+', ' ', page_text)
            
            page_text = re.sub(r'\s*\n\s*\n+\s*', '\n', page_text)
            
            page_text = page_text.replace('\n', ' ')
            
            page_text = re.sub(r' +', ' ', page_text).strip()

            if not page_text: 
                continue

            sentences_on_page = sent_tokenize(page_text)
            
            page_sentence_idx = 0
            for sentence_text in sentences_on_page:
                sentence_text = sentence_text.strip()

                if len(sentence_text.split()) < 2:
                    continue
                if not re.search(r'[a-zA-Z]', sentence_text):
                    continue
                if re.fullmatch(r'[\d\s\.]+', sentence_text) and len(sentence_text) < 20:
                    continue
                if (sentence_text.lower().startswith("figure ") or \
                    sentence_text.lower().startswith("table ") or \
                    sentence_text.lower().startswith("fig. ")) and \
                   len(sentence_text.split()) < 5: 
                     continue


                doc_id = f"sentence_{sentence_counter}"
                self.doc_texts[doc_id] = sentence_text
                self.doc_metadata[doc_id] = {
                    "original_doc": self.current_doc_name,
                    "page_num": page_num,
                    "sentence_idx_in_page": page_sentence_idx # Index of this valid sentence within the current page's extracted valid sentences
                }
                all_sentences_data.append((doc_id, sentence_text))
                sentence_counter += 1
                page_sentence_idx += 1
        
        print(f"DEBUG: Parsed {len(all_sentences_data)} sentences from PDF with new logic.")
        return all_sentences_data

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

    def build_index(self, documents): # 'documents' is all_sentences_data
        self.total_docs = len(documents)
        for doc_id, text in documents:
            tokens = self.preprocess_text(doc_id, text)
            # DEBUG: Print tokens for a few documents to see if they are empty
            # if self.total_docs > 0 and (list(self.doc_texts.keys()).index(doc_id) < 5 or list(self.doc_texts.keys()).index(doc_id) > self.total_docs - 5):
            #     print(f"DEBUG build_index: tokens for doc '{doc_id}': {tokens[:10]}...")

            # Corrected doc_freq logic: it should count if a term appears in a document at all.
            # This means we update doc_freq for a term only once per document.
            # A simple way is to track terms seen for the current doc_id during token iteration.
            terms_in_current_doc = set()
            for pos, token in enumerate(tokens):
                self.inverted_index[token][doc_id].append(pos)
                terms_in_current_doc.add(token)
            
            for term in terms_in_current_doc:
                self.doc_freq[term] += 1
        
        # Final check on doc_freq
        # for term, freq in self.doc_freq.items():
        #     if freq == 0:
        #         print(f"WARNING: Term '{term}' has zero doc_freq.")


    def format_chunk(self, doc_id, score=None, query_method=None):
        """Create a standardized chunk dictionary with metadata."""
        # Ensure metadata exists for the doc_id, providing defaults if not
        metadata = self.doc_metadata.get(doc_id, {
            "original_doc": self.current_doc_name or "Unknown Document",
            "page_num": "unknown",
            "sentence_idx_in_page": "unknown"
        })
        text = self.doc_texts.get(doc_id, "Text not found for this ID.")

        chunk = {
            "doc_id": doc_id,
            "text": text,
            "metadata": {
                "doc": metadata.get("original_doc", self.current_doc_name or "Unknown Document"),
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
        tf = len(self.inverted_index[term][doc_id]) if doc_id in self.inverted_index.get(term, {}) else 0
        df = self.doc_freq.get(term, 0)
        
        # DEBUG: Check values before IDF calculation
        # if tf > 0:
        #     print(f"DEBUG compute_tf_idf: term='{term}', doc_id='{doc_id}', tf={tf}, df={df}, total_docs={self.total_docs}")

        idf = math.log(self.total_docs / (df + 1)) if df > 0 and self.total_docs > 0 else 0 # Added +1 to df for smoothing
        
        # DEBUG: Check final TF-IDF score
        # if tf > 0 and idf > 0:
        #     print(f"DEBUG compute_tf_idf: score={tf * idf}")

        return tf * idf

    # --- Retrieval Methods with refined scoring ---

    def _calculate_base_tfidf_score(self, doc_id, query_tokens): # Changed 'terms_to_score' to 'query_tokens' for clarity
        """Helper to calculate TF-IDF sum for a set of query terms in a document."""
        score = 0.0
        # Ensure doc_id is valid and present in doc_texts before proceeding
        if doc_id not in self.doc_texts:
            # print(f"DEBUG _calculate_base_tfidf_score: doc_id '{doc_id}' not found in doc_texts.")
            return 0.0
            
        for term in query_tokens:
            score += self.compute_tf_idf(term, doc_id)
        return score

    def global_search(self, query_tokens):
        retrieved_chunks = []
        # DEBUG: Check query_tokens for global_search
        # print(f"DEBUG global_search: query_tokens={query_tokens}")
        
        for doc_id in self.doc_texts.keys(): # Iterate through known doc_ids
            score = self._calculate_base_tfidf_score(doc_id, query_tokens)
            if score > 0:
                chunk = self.format_chunk(doc_id, score, "global")
                retrieved_chunks.append(chunk)
        return retrieved_chunks

    def top_k_search(self, query_tokens, k=20):
        doc_scores = defaultdict(float)
        # DEBUG: Check query_tokens for top_k_search
        # print(f"DEBUG top_k_search: query_tokens={query_tokens}")

        for doc_id in self.doc_texts.keys(): # Iterate through known doc_ids
            score = self._calculate_base_tfidf_score(doc_id, query_tokens)
            if score > 0: # Only consider documents with a positive score
                doc_scores[doc_id] = score

        # Using a min-heap to find top K scores efficiently
        # Store as (score, doc_id) to use score for heap comparison
        min_heap = []
        for doc_id, score in doc_scores.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (score, doc_id))
            else:
                # If current score is greater than the smallest score in heap, replace it
                if score > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (score, doc_id))
        
        retrieved_chunks = []
        # Heap contains (score, doc_id), extract and format
        # Sort by score descending for final output
        sorted_heap_items = sorted(min_heap, key=lambda x: x[0], reverse=True)
        for score, doc_id in sorted_heap_items:
            chunk = self.format_chunk(doc_id, score, "top_k")
            retrieved_chunks.append(chunk)
        return retrieved_chunks


    def phrase_query(self, query_raw_tokens):
        results = []
        if not query_raw_tokens:
            # print("DEBUG phrase_query: query_raw_tokens is empty.")
            return []

        stemmed_query_tokens = [self.stemmer.stem(token.lower()) for token in query_raw_tokens if token.isalnum()]
        # print(f"DEBUG phrase_query: stemmed_query_tokens={stemmed_query_tokens}")

        if not stemmed_query_tokens or len(stemmed_query_tokens) < 2 : # Phrase query needs at least 2 terms
            # print("DEBUG phrase_query: stemmed_query_tokens is empty or less than 2 terms after processing.")
            return []

        # Check if all terms exist in the inverted index
        for term in stemmed_query_tokens:
            if term not in self.inverted_index:
                # print(f"DEBUG phrase_query: Term '{term}' not found in inverted index. Phrase not possible.")
                return []
        
        # Get document sets for each term
        doc_sets = [set(self.inverted_index[term].keys()) for term in stemmed_query_tokens]
        # Find common documents that contain all terms of the phrase
        candidate_docs = set.intersection(*doc_sets)
        # print(f"DEBUG phrase_query: Found {len(candidate_docs)} candidate documents for phrase: {stemmed_query_tokens}")

        for doc_id in candidate_docs:
            # Get positions list for each term in the current document
            # positions_for_terms_in_doc = [self.inverted_index[term].get(doc_id, []) for term in stemmed_query_tokens]
            
            # Ensure all terms are actually in this doc_id (intersection might pass docs where a term was removed by other logic)
            term_positions_in_doc = []
            valid_doc_for_all_terms = True
            for term in stemmed_query_tokens:
                if doc_id not in self.inverted_index[term]:
                    valid_doc_for_all_terms = False
                    break
                term_positions_in_doc.append(self.inverted_index[term][doc_id])
            
            if not valid_doc_for_all_terms:
                continue

            first_term_positions = term_positions_in_doc[0]
            for start_pos in first_term_positions:
                is_phrase_match = True
                for i, subsequent_term_positions in enumerate(term_positions_in_doc[1:], 1):
                    # Check if (start_pos + i) exists in the positions of the i-th term (after the first)
                    if (start_pos + i) not in subsequent_term_positions:
                        is_phrase_match = False
                        break
                if is_phrase_match:
                    phrase_score = self._calculate_base_tfidf_score(doc_id, stemmed_query_tokens)
                    phrase_score += 7.0 # Bonus for phrase match
                    chunk = self.format_chunk(doc_id, phrase_score, "phrase")
                    results.append(chunk)
                    break # Found phrase in this doc, move to next candidate doc
        return results

    def wildcard_query(self, query_string):
        # Assuming query_string contains the wildcard, e.g., "comp*t"
        # Avoid stemming the wildcard pattern itself.
        pattern_str = query_string.lower().replace("*", ".*").replace("?", ".") # Basic wildcard handling
        # print(f"DEBUG wildcard_query: pattern_str='{pattern_str}'")
        try:
            pattern = re.compile(pattern_str)
        except re.error:
            # print(f"DEBUG wildcard_query: Invalid regex pattern: {pattern_str}")
            return []

        retrieved_chunks = []
        doc_scores = defaultdict(float)

        matched_terms_from_index = set()
        for term_in_index in self.inverted_index.keys(): # Iterate through terms in the index
            if pattern.fullmatch(term_in_index): # Use fullmatch to ensure the whole term matches the pattern
                matched_terms_from_index.add(term_in_index)
        
        # print(f"DEBUG wildcard_query: Matched terms from index: {list(matched_terms_from_index)[:10]}...")
        if not matched_terms_from_index:
            return []

        for matched_term in matched_terms_from_index:
            for doc_id in self.inverted_index[matched_term].keys():
                doc_scores[doc_id] += self.compute_tf_idf(matched_term, doc_id)

        for doc_id, score in doc_scores.items():
            if score > 0:
                final_score = score + 2.0 # Bonus for wildcard match
                chunk = self.format_chunk(doc_id, final_score, "wildcard")
                retrieved_chunks.append(chunk)
        return retrieved_chunks


    def boolean_query(self, query_string):
        # Basic AND/OR/NOT. For simplicity, assumes terms are space-separated.
        # Example: "apple AND banana", "apple OR orange", "apple NOT orange"
        # This implementation will be basic and handle one operator type per query for simplicity.
        # It will stem the terms involved in the boolean query.
        
        parts = query_string.lower().split()
        retrieved_chunks = []
        
        operator = None
        if "and" in parts:
            operator = "and"
        elif "or" in parts:
            operator = "or"
        elif "not" in parts: # NOT is more complex (requires a base set to subtract from)
            operator = "not"
            # For a simple NOT A, it means docs not containing A.
            # For A NOT B, it means docs with A but not B. We'll handle simpler "NOT term" for now.

        terms_raw = [p for p in parts if p not in ["and", "or", "not"]]
        if not terms_raw:
            # print("DEBUG boolean_query: No terms provided for boolean operation.")
            return []
            
        stemmed_terms = [self.stemmer.stem(t) for t in terms_raw if t.isalnum()]
        # print(f"DEBUG boolean_query: operator='{operator}', stemmed_terms={stemmed_terms}")

        if not stemmed_terms:
            # print("DEBUG boolean_query: stemmed_terms is empty after processing.")
            return []

        doc_ids_to_score = set()

        if operator == "and":
            if not stemmed_terms: return []
            doc_sets = [set(self.inverted_index.get(term, {}).keys()) for term in stemmed_terms]
            if not all(doc_sets): # If any term is not in index, intersection is empty
                # print(f"DEBUG boolean_query (AND): One or more terms not found or yield no docs: {stemmed_terms}")
                return []
            doc_ids_to_score = set.intersection(*doc_sets)
            # print(f"DEBUG boolean_query (AND): Found {len(doc_ids_to_score)} common documents.")
        
        elif operator == "or":
            if not stemmed_terms: return []
            for term in stemmed_terms:
                doc_ids_to_score.update(self.inverted_index.get(term, {}).keys())
            # print(f"DEBUG boolean_query (OR): Found {len(doc_ids_to_score)} combined documents.")

        elif operator == "not": # term1 NOT term2 (docs with term1 but not term2) or just NOT term2 (docs without term2)
            if len(stemmed_terms) == 1: # NOT termA
                term_to_exclude = stemmed_terms[0]
                docs_with_term = set(self.inverted_index.get(term_to_exclude, {}).keys())
                all_docs_in_collection = set(self.doc_texts.keys())
                doc_ids_to_score = all_docs_in_collection - docs_with_term
            elif len(stemmed_terms) == 2 : # termA NOT termB
                term_A_docs = set(self.inverted_index.get(stemmed_terms[0], {}).keys())
                term_B_docs = set(self.inverted_index.get(stemmed_terms[1], {}).keys())
                doc_ids_to_score = term_A_docs - term_B_docs
            else: # Unsupported NOT query structure
                # print("DEBUG boolean_query (NOT): Unsupported NOT query structure. Use 'termA NOT termB' or 'NOT termA'.")
                return []
            # print(f"DEBUG boolean_query (NOT): Found {len(doc_ids_to_score)} documents after NOT operation.")

        else: # No explicit boolean operator found, treat as global search on the terms
            # print("DEBUG boolean_query: No explicit boolean operator, falling back to global search logic.")
            return self.global_search(stemmed_terms) # Use global search for implicit AND or if only one term

        bonus_score = {"and": 6.0, "or": 3.0, "not": 4.0}.get(operator, 0)
        for doc_id in doc_ids_to_score:
            # Score based on all terms if AND/OR, or primary term if "A NOT B"
            score_terms = stemmed_terms
            if operator == "not" and len(stemmed_terms) == 2: # For "A NOT B", score based on A
                score_terms = [stemmed_terms[0]]
            elif operator == "not" and len(stemmed_terms) == 1: # For "NOT A", TF-IDF is not directly applicable. Maybe a fixed score or 0.
                                                                # Let's assign a small constant score for presence in this set.
                chunk = self.format_chunk(doc_id, bonus_score, "boolean_not") # Special method name for NOT
                retrieved_chunks.append(chunk)
                continue


            score = self._calculate_base_tfidf_score(doc_id, score_terms)
            if score > 0 or operator == "not": # For NOT, score might be 0 but doc is still relevant by exclusion
                final_score = score + bonus_score
                chunk = self.format_chunk(doc_id, final_score, f"boolean_{operator}")
                retrieved_chunks.append(chunk)
                
        return retrieved_chunks

    def synonym_query(self, query_raw_tokens):
        expanded_terms_stemmed = set()
        # Stem original query tokens first
        original_stemmed = {self.stemmer.stem(token.lower()) for token in query_raw_tokens if token.isalnum()}
        expanded_terms_stemmed.update(original_stemmed)

        for token in query_raw_tokens: # Use raw tokens for WordNet lookup
            if not token.isalnum(): continue # Skip punctuation for WordNet
            synsets = wordnet.synsets(token.lower())
            for syn in synsets:
                for lemma in syn.lemmas():
                    # Stem the synonyms before adding
                    stemmed_lemma = self.stemmer.stem(lemma.name().lower().replace('_', ' ')) # Handle multi-word lemmas
                    if stemmed_lemma.isalnum(): # Check again after stemming
                       expanded_terms_stemmed.add(stemmed_lemma)
        
        final_expanded_terms_list = list(expanded_terms_stemmed)
        # print(f"DEBUG synonym_query: Final expanded terms ({len(final_expanded_terms_list)}): {final_expanded_terms_list[:15]}...")

        if not final_expanded_terms_list:
            return []

        retrieved_chunks = []
        doc_scores = defaultdict(float)

        # Score documents based on the expanded set of terms (TF-IDF sum)
        for doc_id in self.doc_texts.keys():
            score = self._calculate_base_tfidf_score(doc_id, final_expanded_terms_list)
            if score > 0:
                doc_scores[doc_id] = score
        
        for doc_id, score in doc_scores.items():
            final_score = score + 1.0 # Bonus for synonym expansion match
            chunk = self.format_chunk(doc_id, final_score, "synonym")
            retrieved_chunks.append(chunk)
            
        return retrieved_chunks


    def retrieve_and_score_all(self, query_string, top_n=5):
        # print(f"DEBUG retrieve_and_score_all: Incoming query_string='{query_string}'")
        
        # Preprocess query_string once for methods needing stemmed tokens
        processed_query_tokens = self.preprocess_text("query", query_string) # For global, top_k
        # Get raw tokens for methods like phrase, synonym, or wildcard (if it's not stemmed there)
        raw_query_tokens = word_tokenize(query_string) # For phrase, synonym

        # Check if preprocessing yields anything
        if not processed_query_tokens and not query_string.strip(): # query_string.strip() in case it's just spaces
            # print("DEBUG retrieve_and_score_all: Query tokens are empty after preprocessing and raw query is blank. Returning empty list.")
            return []

        all_results_map = {} # Use a map {doc_id: chunk_data} to easily update/override with better scores

        # --- Global Search (basic TF-IDF sum) ---
        if processed_query_tokens:
            # print(f"Running global search for tokens: {processed_query_tokens}")
            results_global = self.global_search(processed_query_tokens)
            self.add_results_to_map(all_results_map, results_global)
            # print(f"Global search added {len(results_global)} chunks. Current unique results: {len(all_results_map)}")

        # --- Top-K Search (more refined TF-IDF, uses same tokens as global) ---
        if processed_query_tokens:
            # print(f"Running top_k search for tokens: {processed_query_tokens}")
            results_top_k = self.top_k_search(processed_query_tokens, k=20) # Retrieve more for potential diversity
            self.add_results_to_map(all_results_map, results_top_k)
            # print(f"Top-K search added {len(results_top_k)} chunks. Current unique results: {len(all_results_map)}")

        # --- Phrase Query ---
        # print(f"Running phrase query for raw tokens: {raw_query_tokens}")
        if len(raw_query_tokens) > 1: # Phrase query needs more than one token
             results_phrase = self.phrase_query(raw_query_tokens) # Uses raw tokens, stems inside
             self.add_results_to_map(all_results_map, results_phrase)
            #  print(f"Phrase query added {len(results_phrase)} chunks. Current unique results: {len(all_results_map)}")
        # else:
            #  print("Skipping phrase query (single token or no tokens).")

        # --- Wildcard Query ---
        # print(f"Running wildcard query for string: {query_string}")
        if "*" in query_string or "?" in query_string: # Check for wildcard characters
            results_wildcard = self.wildcard_query(query_string) # Uses raw query string
            self.add_results_to_map(all_results_map, results_wildcard)
            # print(f"Wildcard query added {len(results_wildcard)} chunks. Current unique results: {len(all_results_map)}")
        # else:
            # print("Skipping wildcard query (no '*' or '?' found).")
            
        # --- Boolean Query ---
        # print(f"Running boolean query for string: {query_string}")
        query_lower = query_string.lower()
        if " and " in query_lower or " or " in query_lower or " not " in query_lower or \
           query_lower.startswith("not "):
            results_boolean = self.boolean_query(query_string) # Uses raw query string, tokenizes and stems inside
            self.add_results_to_map(all_results_map, results_boolean)
            # print(f"Boolean query added {len(results_boolean)} chunks. Current unique results: {len(all_results_map)}")
        # else:
            # print("Skipping boolean query (no boolean operators detected by simple check).")

        # --- Synonym Query ---
        if raw_query_tokens: # Needs tokens to find synonyms for
            # print(f"Running synonym query for raw tokens: {raw_query_tokens}")
            results_synonym = self.synonym_query(raw_query_tokens) # Uses raw tokens, stems inside
            self.add_results_to_map(all_results_map, results_synonym)
            # print(f"Synonym query added {len(results_synonym)} chunks. Current unique results: {len(all_results_map)}")
        # else:
            # print("Skipping synonym query (no raw tokens).")


        # Sort all unique chunks by score in descending order
        # The values of all_results_map are the chunk dictionaries
        sorted_results = sorted(list(all_results_map.values()), key=lambda x: x["score"], reverse=True)
        # print(f"DEBUG retrieve_and_score_all: Total unique chunks before slicing: {len(sorted_results)}")
        # print(f"DEBUG retrieve_and_score_all: Top {top_n} results scores: {[chunk['score'] for chunk in sorted_results[:top_n]]}")


        return sorted_results[:top_n]

    def add_results_to_map(self, all_results_dict, new_results_list): # Renamed for clarity
        for chunk in new_results_list:
            doc_id = chunk["doc_id"]
            new_score = chunk.get("score", 0.0)
            
            # If doc_id not in dict, or new_score is higher, update/add the chunk.
            # This also means if a chunk was found by multiple methods, the one giving a higher score (due to bonus) wins.
            if doc_id not in all_results_dict or new_score > all_results_dict[doc_id]["score"]:
                all_results_dict[doc_id] = chunk # chunk already contains all necessary fields from format_chunk

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