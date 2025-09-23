    from flask import Flask, request, jsonify, render_template
    from flask_cors import CORS
    import heapq
    import math

    # campus data (graph edges)
    raw_graph = {
        "Gate 1": {"Gate 2": 180},
        "Gate 2": {"Admin block 1": 75, "Parents stay area": 283},
        "Admin block 1": {"Admin block 2": 125, "Parents stay area": 250},
        "Admin block 2": {"Food court": 215, "Staff quarters": 470, "Parents stay area": 305},
        "Food court": {"Parents stay area": 215, "Hostel-1": 271},
        "Hostel-1": {"Admin block 2": 438, "Sport area": 565},
        "Parents stay area": {"Food court": 415, "Hostel-1": 645},
    }

    # Build undirected graph
    campus_graph = {}
    for a, neighbors in raw_graph.items():
        campus_graph.setdefault(a, {})
        for b, d in neighbors.items():
            campus_graph[a][b] = d
            campus_graph.setdefault(b, {})
            if a not in campus_graph[b]:
                campus_graph[b][a] = d

    # Example coordinates (replace with real campus map points)
    coordinates = {
        "Gate 1": [12.9350, 77.6100],
        "Gate 2": [12.9352, 77.6110],
        "Admin block 1": [12.9358, 77.6115],
        "Admin block 2": [12.9365, 77.6120],
        "Parents stay area": [12.9360, 77.6135],
        "Food court": [12.9370, 77.6140],
        "Hostel-1": [12.9380, 77.6150],
        "Staff quarters": [12.9368, 77.6160],
        "Sport area": [12.9390, 77.6170],
    }

    # Normalize names
    def normalize_name(s: str) -> str:
        return ''.join(ch.lower() for ch in s if ch.isalnum())

    all_nodes = sorted(set(campus_graph.keys()))
    norm_map = {normalize_name(name): name for name in all_nodes}

    def lookup_node(user_input: str):
        key = normalize_name(user_input)
        return norm_map.get(key)

    # UCS shortest path
    def ucs(start, goal):
        if start == goal:
            return [start], 0
        pq = [(0, start, [start])]
        visited_cost = {}
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal:
                return path, cost
            if node in visited_cost and cost > visited_cost[node]:
                continue
            visited_cost[node] = cost
            for neighbor, dist in campus_graph.get(node, {}).items():
                new_cost = cost + dist
                if neighbor not in visited_cost or new_cost < visited_cost.get(neighbor, math.inf):
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
        return None, math.inf

    # Flask app
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def home():
        return render_template("ai2.html")

    @app.route("/route", methods=["POST"])
    def find_route():
        data = request.get_json()
        src = lookup_node(data.get("source", ""))
        dst = lookup_node(data.get("destination", ""))

        if src is None:
            return jsonify({"error": f"Unknown location: {data.get('source')}"}), 400
        if dst is None:
            return jsonify({"error": f"Unknown location: {data.get('destination')}"}), 400

        path, cost = ucs(src, dst)
        if path:
            coords = [coordinates[node] for node in path if node in coordinates]
            return jsonify({"path": path, "coords": coords, "distance": cost})
        else:
            return jsonify({"error": f"No path found from {src} to {dst}"}), 404

    if __name__ == "__main__":
        app.run(debug=True)
