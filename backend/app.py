import os
from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename
from InfoRetrSys import InfoRetrievalSystem
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'KFC_V_ME_50'
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

retrieval_system = InfoRetrievalSystem()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            retrieval_system.process_pdf(filepath)
            session['processed_pdf_path'] = filepath
            print(f"PDF processed and session set: {session.get('processed_pdf_path')}")
            return jsonify({'message': f'PDF "{filename}" processed successfully!', 'success': True}), 200
        except Exception as e:
            print(f"Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing PDF: {str(e)}', 'success': False}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDFs are allowed.', 'success': False}), 400

@app.route('/query', methods=['POST'])
def query_pdf():
    if 'processed_pdf_path' not in session:
        print("Session 'processed_pdf_path' not found.")
        return jsonify({'error': 'No PDF has been processed yet. Please upload a PDF first.'}), 400

    data = request.get_json()
    user_query = data.get('query')
    use_llm = data.get('use_llm', False)
    top_n = data.get('top_n', 5)
    scoring_method = data.get('scoring_method', 'tf_idf')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        top_chunks, other_method_chunks = retrieval_system.retrieve_and_score_all(user_query, top_n=top_n, scoring_method=scoring_method)

        answer = ""
        if use_llm:
            if top_chunks:
                answer = retrieval_system.generate_answer(user_query, top_chunks)
            else:
                answer = "No relevant information found for LLM to generate an answer."
        else:
            answer = "LLM generation is off. Showing top relevant chunks."

        formatted_top_chunks = [
            {
                'text': chunk['text_preview'],
                'original_text': chunk['text'],
                'doc': chunk['metadata'].get('doc', 'unknown'),
                'page': chunk['metadata'].get('page', 'unknown'),
                'sentence_index': chunk['metadata'].get('sentence_idx_in_page', 'N/A'),
                'score': round(chunk.get('score', 0.0), 4),
                'query_methods': chunk.get('query_methods', ['N/A'])
            } for chunk in top_chunks
        ]

        formatted_other_method_chunks = [
            {
                'text': chunk['text_preview'],
                'original_text': chunk['text'],
                'doc': chunk['metadata'].get('doc', 'unknown'),
                'page': chunk['metadata'].get('page', 'unknown'),
                'sentence_index': chunk['metadata'].get('sentence_idx_in_page', 'N/A'),
                'score': round(chunk.get('score', 0.0), 4),
                'query_methods': chunk.get('query_methods', ['N/A'])
            } for chunk in other_method_chunks
        ]

        return jsonify({
            'answer': answer,
            'chunks': formatted_top_chunks,
            'other_method_chunks': formatted_other_method_chunks
        }), 200
    except Exception as e:
        print(f"Error during query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error retrieving answer: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)