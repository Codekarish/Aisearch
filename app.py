from flask import Flask, request, jsonify, render_template
from main_last import get_q, optimized_faiss_search, refined_sorting_logic, properties, index, tokenizer, model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('aisearch.html', properties=None)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get("query", "")
    print(f"User Query: {query}")

    # Process the query using get_q function
    processed_query = get_q(query)
    print(f"Processed Query: {processed_query}")

    # Perform the search
    topk = int(request.form.get('k', 20))
    search_result = optimized_faiss_search(processed_query, index, tokenizer, model, properties, topk=topk, nprobe=5)

    # Drop unnecessary columns
    search_result = search_result.drop(['description_embedding', 'location_embedding', 'bedroom_embedding', 'submission_type_embedding', 'property_type_embedding', 'price_embedding', 'agent_embedding'], axis=1, errors='ignore')

    if not search_result.empty:
        return render_template('aisearch.html', properties=search_result)
    else:
        return render_template('aisearch.html', properties=None, error="No matching results found.")

if __name__ == '__main__':
    app.run(debug=True)
