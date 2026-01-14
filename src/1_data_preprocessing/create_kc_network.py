import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import pandas as pd
import matplotlib.font_manager as fm

class KCNetworkBuilder:
    def __init__(self, questions_path, kc_map_path, kc_map_en_path):
        self.questions_path = questions_path
        self.kc_map_path = kc_map_path
        self.kc_map_en_path = kc_map_en_path
        self.questions = None
        self.kc_map = None  # Chinese → KC ID
        self.kc_map_en = None  # KC ID → English
        self.graph = None
        self.shortest_paths = None
        
        self.setup_fonts()
        
        self.load_data()
    
    def setup_fonts(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("English font setup completed: DejaVu Sans")
        
    def load_data(self):
        try:
            with open(self.questions_path, 'r', encoding='utf-8-sig') as f:
                self.questions = json.load(f)
            print(f"Questions data loaded successfully: {len(self.questions)} questions")
            
            with open(self.kc_map_path, 'r', encoding='utf-8') as f:
                self.kc_map = json.load(f)
            print(f"KC mapping data loaded successfully: {len(self.kc_map)} KCs")
            
            with open(self.kc_map_en_path, 'r', encoding='utf-8') as f:
                self.kc_map_en = json.load(f)
            print(f"KC English mapping data loaded successfully: {len(self.kc_map_en)} KCs")
            
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            raise
    
    def get_kc_name_by_id(self, kc_id):
        if kc_id.isdigit():
            return self.kc_map_en.get(kc_id, f"KC_{kc_id}")
        
        return kc_id
    
    def convert_chinese_to_english(self, chinese_kc_name):
        kc_id = None
        for k_id, chinese_name in self.kc_map.items():
            if chinese_name == chinese_kc_name:
                kc_id = k_id
                break
        
        if kc_id is None:
            print(f"[Warning] Unmapped Chinese KC: {chinese_kc_name}")
            return chinese_kc_name
        
        english_name = self.kc_map_en.get(kc_id, f"KC_{kc_id}")
        return english_name
    
    def get_kc_id_by_english_name(self, english_name):
        for kc_id, en_name in self.kc_map_en.items():
            if en_name == english_name:
                return kc_id
        
        if english_name.isdigit():
            return english_name
        
        print(f"[Warning] English name not found in mapping: {english_name}")
        return None
    
    def extract_kc_edges(self):
        edges = set()
        edge_counts = defaultdict(int) 
        
        for q_id, q in self.questions.items():
            kc_routes = q.get("kc_routes", [])
            if not kc_routes or not isinstance(kc_routes, list):
                continue
                
            for route in kc_routes:
                if isinstance(route, str) and "----" in route:
                    kc_list = [kc.strip() for kc in route.split("----") if kc.strip()]
                    
                    for i in range(len(kc_list) - 1):
                        source_kc = self.convert_chinese_to_english(kc_list[i])
                        target_kc = self.convert_chinese_to_english(kc_list[i + 1])
                        
                        if source_kc != target_kc: 
                            edges.add((source_kc, target_kc))
                            edge_counts[(source_kc, target_kc)] += 1
        
        print(f"Total {len(edges)} unique directed edges extracted")
        print(f"Most frequent edge: {max(edge_counts.items(), key=lambda x: x[1]) if edge_counts else 'None'}")
        
        return edges, edge_counts
    
    def build_network(self):
        edges, edge_counts = self.extract_kc_edges()
        
        self.graph = nx.DiGraph()
        
        for edge in edges:
            weight = edge_counts[edge]
            self.graph.add_edge(edge[0], edge[1], weight=weight)
        
        for node in self.graph.nodes():
            kc_id = self.get_kc_id_by_english_name(node)
            if kc_id is None:
                print(f"[Error] KC ID not found for node: {node}")
                self.graph.nodes[node]['kc_id'] = f"UNKNOWN_{node}"
                self.graph.nodes[node]['english_name'] = node
            else:
                self.graph.nodes[node]['kc_id'] = kc_id
                self.graph.nodes[node]['english_name'] = node
        
        print(f"Network construction completed:")
        print(f"  - Number of nodes: {self.graph.number_of_nodes()}")
        print(f"  - Number of edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def debug_kc_mapping(self, sample_size=5):
        print("\n=== KC Mapping Debug ===")
        
        print(f"KC mapping sample (first {sample_size}):")
        for i, (kc_id, en_name) in enumerate(list(self.kc_map_en.items())[:sample_size]):
            print(f"  KC ID: {kc_id} -> English: {en_name}")
        
        print(f"\nReverse mapping test (first {sample_size}):")
        for i, (kc_id, en_name) in enumerate(list(self.kc_map_en.items())[:sample_size]):
            found_kc_id = self.get_kc_id_by_english_name(en_name)
            print(f"  English: {en_name[:50]}... -> KC ID: {found_kc_id}")
        
        if self.graph is not None:
            print(f"\nNetwork nodes sample (first {sample_size}):")
            for i, node in enumerate(list(self.graph.nodes())[:sample_size]):
                kc_id = self.graph.nodes[node]['kc_id']
                print(f"  Node: {node[:50]}... -> KC ID: {kc_id}")
    

    def shortest_path(self):
        source = []
        target = []
        shortest_path = []
        shortest_path_len = []

        undirected_g = self.graph.to_undirected()

        for s in self.graph:
            for t in self.graph:
                spath = nx.dijkstra_path(undirected_g, s, t, weight='weight')
                source.append(s)
                target.append(t)
                shortest_path.append(spath)
                shortest_path_len.append(len(spath))

        df = pd.DataFrame({'source':source, 'target':target, 'shortest_path':shortest_path, 'shortest_path_len':shortest_path_len})
        df_np = df.to_numpy()

        # Build a lookup dictionary
        self.shortest_paths = { (row[0], row[1]): row[3] for row in df_np }

        return self.shortest_paths

    
    
    def save_network(self, output_dir="./data/XES3G5M/metadata/"):
        if self.graph is None:
            print("Please build the network first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save KC relation graph
        pickle_path = os.path.join(output_dir, "kc_network.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Network saved in pickle format: {pickle_path}")
        
        
        # Save node information
        nodes_csv_path = os.path.join(output_dir, "kc_nodes.csv")
        with open(nodes_csv_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["node_id", "node_name", "in_degree", "out_degree", "total_degree"])
            for node in self.graph.nodes():
                in_deg = self.graph.in_degree(node)
                out_deg = self.graph.out_degree(node)
                total_deg = in_deg + out_deg
                kc_id = self.graph.nodes[node]['kc_id'] 
                writer.writerow([kc_id, node, in_deg, out_deg, total_deg])
        print(f"Node list saved as CSV: {nodes_csv_path}")

        # save shortest paths lookup dictionary

        shortest_save_dir = os.path.join(output_dir, "shortest_paths.pkl")
        with open(shortest_save_dir, 'wb') as f:
            pickle.dump(self.shortest_paths, f)
        print(f"Shortest paths saved in pickle format: {shortest_save_dir}")
        
        # Save network stats
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_connected": nx.is_weakly_connected(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "components": nx.number_weakly_connected_components(self.graph),
            "density": nx.density(self.graph),
            "average_clustering": nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0
        }
        
        os.makedirs("./results/", exist_ok=True)
        stats_path = "./results/KC_relation_graph_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Network statistics saved as JSON: {stats_path}")

def main():
    questions_path = "./data/XES3G5M/metadata/questions.json"
    kc_map_path = "./data/XES3G5M/metadata/kc_routes_map.json"
    kc_map_en_path = "./data/XES3G5M/metadata/kc_routes_map_en.json"
    
    builder = KCNetworkBuilder(questions_path, kc_map_path, kc_map_en_path)
    
    builder.debug_kc_mapping()
    
    print("\n=== KC Network Construction Started ===")
    graph = builder.build_network()
    shortest = builder.shortest_path()
    
    print("\n=== Network Saving ===")
    builder.save_network()
    
    print("\n=== Completed ===")
    print("Network has been successfully created and saved.")

if __name__ == "__main__":
    main() 