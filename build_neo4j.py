import os
import json
from py2neo import Graph, Node
from tqdm import tqdm


class MedicalGraphBuilder:
    def __init__(self):
        # ==========================================
        # 【重要】请修改这里的密码为你自己的 Neo4j 密码
        # ==========================================
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "Ljy449155760"  # <--- 记得改密码！

        try:
            self.graph = Graph(self.uri, auth=(self.user, self.password))
            print("✅ 成功连接到 Neo4j 数据库")
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            exit()

    def create_indexes(self):
        """【核心优化】创建索引，大幅提升写入速度"""
        print("⚡️ 正在创建索引 (Index)...")
        labels = ['Disease', 'Drug', 'Food', 'Check', 'Department', 'Producer', 'Symptom']

        for label in labels:
            # 创建唯一约束（Unique Constraint），它同时也是一个索引
            # 语法适配 Neo4j 4.x 和 5.x
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
            try:
                self.graph.run(query)
            except Exception as e:
                # 兼容旧版本语法
                try:
                    self.graph.run(f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.name IS UNIQUE")
                except:
                    pass
        print("✅ 索引创建完成！")

    def read_nodes(self):
        print("正在读取 medical.json ...")

        drugs = set()
        foods = set()
        checks = set()
        departments = set()
        producers = set()
        symptoms = set()
        diseases = set()

        disease_infos = []

        # 关系列表
        rels_department = []
        rels_noteat = []
        rels_doeat = []
        rels_recommandeat = []
        rels_commondrug = []
        rels_recommanddrug = []
        rels_check = []
        rels_symptom = []
        rels_acompany = []
        rels_category = []

        count = 0
        file_path = 'medical.json'

        if not os.path.exists(file_path):
            print(f"❌ 错误：找不到文件 {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="解析数据"):
                try:
                    data_json = json.loads(line)
                    disease_dict = {}
                    count += 1

                    disease_name = data_json.get('name', '')
                    if not disease_name: continue

                    disease_dict['name'] = disease_name
                    disease_dict['desc'] = data_json.get('desc', '')
                    disease_dict['prevent'] = data_json.get('prevent', '')
                    disease_dict['cause'] = data_json.get('cause', '')
                    disease_dict['yixue'] = data_json.get('yixue', '')
                    disease_dict['cure_way'] = data_json.get('cure_way', [])

                    diseases.add(disease_name)
                    disease_infos.append(disease_dict)

                    # 提取关联
                    if 'symptom' in data_json:
                        symptoms.update(data_json['symptom'])
                        for symptom in data_json['symptom']:
                            rels_symptom.append([disease_name, symptom])

                    if 'acompany' in data_json:
                        for acompany in data_json['acompany']:
                            diseases.add(acompany)
                            rels_acompany.append([disease_name, acompany])

                    if 'recommand_eat' in data_json:
                        foods.update(data_json['recommand_eat'])
                        for eat in data_json['recommand_eat']:
                            rels_recommandeat.append([disease_name, eat])

                    if 'not_eat' in data_json:
                        foods.update(data_json['not_eat'])
                        for eat in data_json['not_eat']:
                            rels_noteat.append([disease_name, eat])

                    if 'do_eat' in data_json:
                        foods.update(data_json['do_eat'])
                        for eat in data_json['do_eat']:
                            rels_doeat.append([disease_name, eat])

                    if 'recommand_drug' in data_json:
                        drugs.update(data_json['recommand_drug'])
                        for drug in data_json['recommand_drug']:
                            rels_recommanddrug.append([disease_name, drug])

                    if 'common_drug' in data_json:
                        drugs.update(data_json['common_drug'])
                        for drug in data_json['common_drug']:
                            rels_commondrug.append([disease_name, drug])

                    if 'check' in data_json:
                        checks.update(data_json['check'])
                        for check in data_json['check']:
                            rels_check.append([disease_name, check])

                    if 'drug_detail' in data_json:
                        for det in data_json['drug_detail']:
                            det_producer = det.split('(')[0]
                            if det_producer:
                                producers.add(det_producer)

                    if 'cure_department' in data_json:
                        departments.update(data_json['cure_department'])
                        for department in data_json['cure_department']:
                            rels_category.append([disease_name, department])

                except Exception as e:
                    continue

        return (drugs, foods, checks, departments, producers, symptoms, diseases, disease_infos,
                rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department,
                rels_commondrug, rels_symptom, rels_acompany, rels_recommanddrug)

    def clear_graph(self):
        print("正在清空数据库...")
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("✅ 数据库已清空")

    def create_simple_nodes(self, label, nodes_set):
        if not nodes_set: return
        print(f"正在创建 [{label}] 节点，共 {len(nodes_set)} 个...")
        # MERGE 会自动利用我们刚才创建的唯一索引
        query = f"UNWIND $names AS name MERGE (n:{label} {{name: name}})"
        self.graph.run(query, names=list(nodes_set))

    def create_disease_nodes(self, disease_infos):
        print(f"正在创建 [Disease] 节点，共 {len(disease_infos)} 个...")
        query = """
        UNWIND $batch AS data
        MERGE (n:Disease {name: data.name})
        SET n.desc = data.desc, n.prevent = data.prevent, n.cause = data.cause, n.yixue = data.yixue
        """
        batch_size = 1000
        for i in tqdm(range(0, len(disease_infos), batch_size), desc="Writing Diseases"):
            batch = disease_infos[i:i + batch_size]
            self.graph.run(query, batch=batch)

    def create_relationships(self, start_label, end_label, edges, rel_type, rel_desc):
        if not edges: return
        print(f"正在构建关系: ({start_label}) -[{rel_type}]-> ({end_label}) 共 {len(edges)} 条...")

        # 这里的 MATCH 因为有了索引，速度会从 100ms 变成 0.1ms
        query = f"""
        UNWIND $batch AS row
        MATCH (n:{start_label} {{name: row[0]}})
        MATCH (m:{end_label} {{name: row[1]}})
        MERGE (n)-[r:{rel_type}]->(m)
        """

        # 增大 Batch Size 提升吞吐量
        batch_size = 5000
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            self.graph.run(query, batch=batch)

    def build(self):
        data = self.read_nodes()
        if not data: return
        (drugs, foods, checks, departments, producers, symptoms, diseases, disease_infos,
         rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department,
         rels_commondrug, rels_symptom, rels_acompany, rels_recommanddrug) = data

        self.clear_graph()

        # 【关键步骤】先建索引！
        self.create_indexes()

        self.create_simple_nodes('Drug', drugs)
        self.create_simple_nodes('Food', foods)
        self.create_simple_nodes('Check', checks)
        self.create_simple_nodes('Department', departments)
        self.create_simple_nodes('Producer', producers)
        self.create_simple_nodes('Symptom', symptoms)
        self.create_disease_nodes(disease_infos)

        self.create_relationships('Disease', 'Department', rels_department, 'belongs_to', '所属科室')
        self.create_relationships('Disease', 'Food', rels_noteat, 'no_eat', '忌吃')
        self.create_relationships('Disease', 'Food', rels_doeat, 'do_eat', '宜吃')
        self.create_relationships('Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationships('Disease', 'Drug', rels_commondrug, 'common_drug', '常用药品')
        self.create_relationships('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '好评药品')
        self.create_relationships('Disease', 'Check', rels_check, 'need_check', '诊断检查')
        self.create_relationships('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')
        self.create_relationships('Disease', 'Disease', rels_acompany, 'acompany_with', '并发症')

        print("\n🎉 知识图谱构建完成！")


if __name__ == "__main__":
    builder = MedicalGraphBuilder()
    builder.build()