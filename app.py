# app.py

from flask import Flask, request, jsonify, render_template
from main_last import get_q, optimized_faiss_search, refined_sorting_logic, properties, index, tokenizer, model

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('aisearch.html')
@app.route('/search', methods=['POST'])
def search():
    inp = request.json
    query = get_q(inp.get("query"))
    print(query)
    topk = inp.get('k', 20)  # Get 'k' from the request, default to 20 if not provided

    search_result = optimized_faiss_search(query, index, tokenizer, model, properties, topk=topk, nprobe=5)
    # Drop unnecessary columns
    search_result = search_result.drop(['description_embedding', 'location_embedding', 'bedroom_embedding', 'submission_type_embedding', 'property_type_embedding', 'price_embedding', 'agent_embedding'], axis=1)

    if not search_result.empty:
        return search_result.to_json(orient='records')
    else:
        return jsonify({"error": "No matching results found."})
if __name__ == '__main__':
    app.run(debug=True)


