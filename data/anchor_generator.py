import os
import random
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def iou_wh(box1, box2):
    # box = [x_center, y_center, width, height]
    w1, h1 = box1
    w2, h2 = box2

    # 计算交集的宽度和高度
    intersect_width = min(w1, w2)
    intersect_height = min(h1, h2)

    # 计算交集区域的面积
    intersection_area = intersect_width * intersect_height

    # 计算每个框的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算并集的面积
    union_area = area1 + area2 - intersection_area

    # 计算IoU
    return intersection_area / union_area

class KMeans:
    def __init__(self, n_clusters, random_state):
        """
        KMeans 聚类算法

        Args:
            n_clusters (_type_): 聚类簇数
            random_state (_type_): 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def fit(self, data):
        """
        执行 K-means 聚类
        :param data: 输入数据，形状为 (n_samples, n_features)
        :return: None
        """
        n_samples, n_features = data.shape
        
        # 1. 初始化： 随机选择k个数据点作为初始簇中心
        self.cluster_centers_ = data[random.sample(range(n_samples), self.n_clusters)]
        
        for iteration in range(self.random_state):
            # 2. 分配数据点到最近的簇中心
            distance = np.linalg.norm(data[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distance, axis=1)  
            
            # 3. 计算新的簇中心
            new_cluster_centers_ = np.array([data[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])      
            
            # 4. 检查收敛条件
            shift = np.linalg.norm(new_cluster_centers_ - self.cluster_centers_)
            print(f"Iteration {iteration + 1}, shift: {shift:.6f}")
            # if shift < 1e-4:
            #     print("Converged")
            #     break
            
            self.cluster_centers_ = new_cluster_centers_

class KMeansIoU:
    def __init__(self, n_clusters, random_state):
        """
        KMeans 聚类算法

        Args:
            n_clusters (_type_): 聚类簇数
            random_state (_type_): 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def fit(self, data):
        """
        执行 K-means 聚类
        :param data: 输入数据，形状为 (n_samples, n_features)
        :return: None
        """
        n_samples = data.shape[0]
        
        # 1. 初始化： 随机选择k个数据点作为初始簇中心
        self.cluster_centers_ = data[random.sample(range(n_samples), self.n_clusters)]
        
        for iteration in range(self.random_state):
            # 2. 分配数据点到最近的簇中心
            iou_matrix = np.zeros((n_samples, self.n_clusters))
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    iou_matrix[i, j] = iou_wh(data[i], self.cluster_centers_[j])
            self.labels_ = np.argmax(iou_matrix, axis=1)
            
            # 3. 计算新的簇中心
            new_cluster_centers_ = np.zeros_like(self.cluster_centers_)
            for i in range(self.n_clusters):
                points_in_cluster = data[self.labels_ == i]
                if points_in_cluster.shape[0] > 0:
                    new_cluster_centers_[i] = points_in_cluster.mean(axis=0)
            
            # 4. 检查收敛条件
            shift = np.linalg.norm(new_cluster_centers_ - self.cluster_centers_)
            print(f"Iteration {iteration + 1}, shift: {shift:.6f}")
            if shift < 1e-4:
                print("Converged")
                break
            
            self.cluster_centers_ = new_cluster_centers_
         
def load_annotations(xml_folders):
    widths = []
    heights = []
    for xml_folder in xml_folders:
        xml_folder = os.path.join(xml_folder, 'Annotations')
        for xml_file in os.listdir(xml_folder):
            if not xml_file.endswith('.xml'):
                continue
            
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                width = xmax - xmin
                height = ymax - ymin
                
                widths.append(width)
                heights.append(height)
            
    return np.array(list(zip(widths, heights)))

def kmeans_anchors(data, n_clusters=9):
    """
    使用Kmeans聚类得到锚框尺寸

    Args:
        data (_type_): 目标标注框的宽高数据
        n_clusters (int, optional): 聚类中心数目， 默认9.
    Return:
        聚类结果
    """
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=n_clusters, random_state=200)
    kmeans = KMeansIoU(n_clusters=n_clusters, random_state=200)
    kmeans.fit(data)
    anchors = kmeans.cluster_centers_
    
    return anchors, kmeans.labels_

def visualize_clusters(data, anchors, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, label='Data')
    plt.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=100, label='Anchors')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('K-means Clustering for Anchor Boxes')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    # 1. 加载标注数据
    xml_folders = ['CCPD2019\ccpd_base', 'CCPD2019\ccpd_challenge', 'CCPD2019\ccpd_db', 'CCPD2019\ccpd_fn', 'CCPD2019\ccpd_rotate', 
                   'CCPD2019\ccpd_tilt', 'CCPD2019\ccpd_weather', 'CCPD2020\ccpd_green', 'CCPD2021\CCPD_yellow']
    data = load_annotations(xml_folders)
    print(f"Loaded {len(data)} bounding boxes")
    
    # 2. 进行K-means聚类
    n_clusters = 9
    anchors, labels = kmeans_anchors(data, n_clusters)
    anchors = anchors.astype(int)
    
    # 3. 可视化聚类结果
    # visualize_clusters(data, anchors, labels)
    
    # 4. 打印锚框结果
    print("Anchor Boxes for YOLO:")
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_anchors = anchors[sorted_indices]
    print(",".join([f"[{w}, {h}]" for w, h in sorted_anchors]))
    
    """
    Anchor Boxes for Private:
    (18, 33),(23, 45),(27, 58),(38, 59),(33, 80),(52, 81),(39, 111),(63, 125),(113, 170)
    
    Anchor Boxes for Public:
    (28, 111),(43, 76),(35, 116),(41, 125),(48, 137),(58, 157),(70, 205),(101, 312),(180, 370)
    """