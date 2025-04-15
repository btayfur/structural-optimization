#örnek bir gezgin satıcı probleminin kodu.
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from matplotlib.animation import FuncAnimation
from itertools import permutations
import imageio

class TSP:
    def __init__(self, num_cities=10, random_seed=42):
        """
        Gezgin Satıcı Problemi (TSP) sınıfı.
        
        Args:
            num_cities: Şehir sayısı
            random_seed: Rastgele sayı üreteci için tohum değeri
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.num_cities = num_cities
        self.cities = self.generate_cities()
        self.distances = self.calculate_distances()
        
    def generate_cities(self):
        """Rastgele şehir konumları oluşturur."""
        return np.random.rand(self.num_cities, 2)
    
    def calculate_distances(self):
        """Şehirler arası mesafe matrisini hesaplar."""
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    distances[i, j] = np.sqrt(np.sum((self.cities[i] - self.cities[j])**2))
        return distances
    
    def calculate_route_length(self, route):
        """Belirli bir rotanın toplam uzunluğunu hesaplar."""
        total_length = 0
        for i in range(len(route) - 1):
            total_length += self.distances[route[i], route[i+1]]
        # Başlangıç noktasına dönüş
        total_length += self.distances[route[-1], route[0]]
        return total_length
    
    def nearest_neighbor(self, start_city=0, save_dir=None):
        """En yakın komşu sezgisel algoritması."""
        unvisited = list(range(self.num_cities))
        route = [start_city]
        unvisited.remove(start_city)
        
        # Create directory for NN process images if not exists
        if save_dir:
            nn_dir = os.path.join(save_dir, "nn_process")
            if not os.path.exists(nn_dir):
                os.makedirs(nn_dir)
            # Save initial route
            current_distance = 0  # Initial distance
            self.save_route_image(route, 0, current_distance, os.path.join(nn_dir, f"nn_step_0.png"), 
                                 title="Nearest Neighbor: Initial")
        
        step = 1
        while unvisited:
            current_city = route[-1]
            nearest = min(unvisited, key=lambda city: self.distances[current_city, city])
            route.append(nearest)
            unvisited.remove(nearest)
            
            # Save intermediate step
            if save_dir:
                current_distance = self.calculate_route_length(route + [route[0]])  # Include return to start
                self.save_route_image(route, step, current_distance, 
                                     os.path.join(nn_dir, f"nn_step_{step}.png"),
                                     title="Nearest Neighbor: Step")
                step += 1
        
        # Create GIF from NN process images
        if save_dir:
            self.create_improvement_gif(nn_dir, os.path.join(save_dir, "nearest_neighbor_process.gif"))
            
        return route
    
    def two_opt(self, route, max_iterations=1000, save_dir=None):
        """2-opt yerel arama algoritması."""
        best_route = route.copy()
        improved = True
        iteration = 0
        
        # Create directory for 2-opt process images if not exists
        if save_dir:
            two_opt_dir = os.path.join(save_dir, "two_opt_process")
            if not os.path.exists(two_opt_dir):
                os.makedirs(two_opt_dir)
            # Save initial route
            initial_distance = self.calculate_route_length(best_route)
            self.save_route_image(best_route, 0, initial_distance, 
                                 os.path.join(two_opt_dir, f"two_opt_step_0.png"),
                                 title="2-opt: Initial Route")
            improvement_count = 1
        
        while improved and iteration < max_iterations:
            improved = False
            best_distance = self.calculate_route_length(best_route)
            
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue  # Bitişik kenarları atla
                    
                    new_route = best_route.copy()
                    # i ve j arasındaki segmenti ters çevir
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    
                    new_distance = self.calculate_route_length(new_route)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route
                        improved = True
                        
                        # Save image for each improvement
                        if save_dir:
                            self.save_route_image(best_route, iteration+1, best_distance, 
                                                os.path.join(two_opt_dir, f"two_opt_step_{improvement_count}.png"),
                                                title="2-opt: Improved Route")
                            improvement_count += 1
            
            iteration += 1
        
        # Create GIF from 2-opt process images
        if save_dir and os.path.exists(two_opt_dir) and len(os.listdir(two_opt_dir)) > 1:
            self.create_improvement_gif(two_opt_dir, os.path.join(save_dir, "two_opt_process.gif"))
            
        return best_route
    
    def simulated_annealing(self, initial_temp=1000, cooling_rate=0.95, max_iterations=1000, save_dir=None):
        """Tavlama benzetimi algoritması."""
        # Rastgele bir başlangıç rotası oluştur
        current_route = list(range(self.num_cities))
        random.shuffle(current_route)
        
        best_route = current_route.copy()
        current_distance = self.calculate_route_length(current_route)
        best_distance = current_distance
        
        temperature = initial_temp
        iteration_history = [(0, best_distance, best_route.copy())]
        
        # Create directory for improvement images if not exists
        if save_dir:
            improvements_dir = os.path.join(save_dir, "sa_process")
            if not os.path.exists(improvements_dir):
                os.makedirs(improvements_dir)
            # Save initial route
            self.save_route_image(best_route, 0, best_distance, 
                                 os.path.join(improvements_dir, f"sa_step_0.png"),
                                 title="Simulated Annealing: Initial")
            improvement_count = 1
        
        for iteration in range(max_iterations):
            # Rastgele iki şehir seç ve yerlerini değiştir
            i, j = sorted(random.sample(range(self.num_cities), 2))
            
            new_route = current_route.copy()
            new_route[i:j+1] = reversed(new_route[i:j+1])  # 2-opt hareketi
            
            new_distance = self.calculate_route_length(new_route)
            
            # Kabul olasılığını hesapla
            delta = new_distance - current_distance
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_route = new_route
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
                    
                    # Save image for each improvement
                    if save_dir:
                        self.save_route_image(best_route, iteration+1, best_distance, 
                                             os.path.join(improvements_dir, f"sa_step_{improvement_count}.png"),
                                             title="Simulated Annealing: Improved")
                        improvement_count += 1
            
            # Sıcaklığı düşür
            temperature *= cooling_rate
            
            # İlerlemeyi kaydet
            if (iteration + 1) % 10 == 0:
                iteration_history.append((iteration + 1, best_distance, best_route.copy()))
        
        # Create GIF from improvement images
        if save_dir:
            self.create_improvement_gif(improvements_dir, os.path.join(save_dir, "simulated_annealing_process.gif"))
        
        return best_route, iteration_history, best_distance
    
    def save_route_image(self, route, iteration, distance, filename, title=None):
        """Rotayı bir resim olarak kaydeder."""
        plt.figure(figsize=(10, 8))
        
        # Şehirleri çiz
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        
        # Şehir numaralarını ekle
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Rotayı çiz
        for i in range(len(route)):
            j = (i + 1) % len(route)
            try:
                plt.plot([self.cities[route[i], 0], self.cities[route[j if j < len(route) else 0], 0]],
                         [self.cities[route[i], 1], self.cities[route[j if j < len(route) else 0], 1]], 'b-')
            except IndexError:
                # Handle case where route is incomplete
                if i < len(route) - 1:  # If not the last city
                    plt.plot([self.cities[route[i], 0], self.cities[route[i+1], 0]],
                             [self.cities[route[i], 1], self.cities[route[i+1], 1]], 'b-')
        
        # İlk şehri vurgula
        if len(route) > 0:
            plt.scatter(self.cities[route[0], 0], self.cities[route[0], 1], c='green', s=150, zorder=3)
        
        if title:
            plt.title(f"{title} - Iteration: {iteration} - Distance: {distance:.2f}")
        else:
            plt.title(f"Iteration: {iteration} - Distance: {distance:.2f}")
            
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        
        plt.savefig(filename)
        plt.close()
    
    def create_improvement_gif(self, image_dir, output_filename, duration=0.5):
        """İyileştirme resimlerinden bir GIF oluşturur."""
        images = []
        image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for filename in image_files:
            images.append(imageio.imread(filename))
        
        imageio.mimsave(output_filename, images, duration=duration)
        print(f"Improvement GIF saved to {output_filename}")
    
    def visualize_route(self, route, title="TSP Rotası", save_path=None):
        """Rotayı görselleştirir."""
        plt.figure(figsize=(10, 8))
        
        # Şehirleri çiz
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        
        # Şehir numaralarını ekle
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Rotayı çiz
        for i in range(len(route)):
            j = (i + 1) % len(route)
            plt.plot([self.cities[route[i], 0], self.cities[route[j], 0]],
                     [self.cities[route[i], 1], self.cities[route[j], 1]], 'b-')
        
        # İlk şehri vurgula
        plt.scatter(self.cities[route[0], 0], self.cities[route[0], 1], c='green', s=150, zorder=3)
        
        plt.title(f"{title} - Mesafe: {self.calculate_route_length(route):.2f}")
        plt.xlabel("X Koordinatı")
        plt.ylabel("Y Koordinatı")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def animate_optimization(self, iteration_history, save_path=None):
        """Optimizasyon sürecini animasyonla gösterir."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            iteration, distance, route = iteration_history[frame]
            
            # Şehirleri çiz
            ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
            
            # Şehir numaralarını ekle
            for i, (x, y) in enumerate(self.cities):
                ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
            
            # Rotayı çiz
            for i in range(len(route)):
                j = (i + 1) % len(route)
                ax.plot([self.cities[route[i], 0], self.cities[route[j], 0]],
                        [self.cities[route[i], 1], self.cities[route[j], 1]], 'b-')
            
            # İlk şehri vurgula
            ax.scatter(self.cities[route[0], 0], self.cities[route[0], 1], c='green', s=150, zorder=3)
            
            ax.set_title(f"İterasyon: {iteration} - Mesafe: {distance:.2f}")
            ax.set_xlabel("X Koordinatı")
            ax.set_ylabel("Y Koordinatı")
            ax.grid(True)
            
            return ax,
        
        ani = FuncAnimation(fig, update, frames=len(iteration_history), interval=200, blit=False)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=5)
        
        plt.show()

# Ana program
if __name__ == "__main__":
    # Visualizations klasörünü oluştur
    # Kodun bulunduğu klasörün içinde visualizations alt klasörünü oluştur
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(current_dir, "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Results dictionary to store algorithm performances
    results = {}
    
    # TSP örneği oluştur
    num_cities = 15
    tsp = TSP(num_cities=num_cities)
    
    # En yakın komşu algoritması
    print("Running Nearest Neighbor algorithm...")
    nn_route = tsp.nearest_neighbor(save_dir=vis_dir)
    nn_distance = tsp.calculate_route_length(nn_route)
    results["Nearest Neighbor"] = {"distance": nn_distance, "route": nn_route}
    print(f"Nearest Neighbor algorithm distance: {nn_distance:.2f}")
    tsp.visualize_route(nn_route, "Nearest Neighbor Algorithm", 
                        save_path=os.path.join(vis_dir, "nearest_neighbor.png"))
    
    # 2-opt iyileştirmesi
    print("Running 2-opt improvement...")
    improved_route = tsp.two_opt(nn_route, save_dir=vis_dir)
    improved_distance = tsp.calculate_route_length(improved_route)
    results["2-opt"] = {"distance": improved_distance, "route": improved_route, "improvement": nn_distance - improved_distance}
    print(f"2-opt improvement distance: {improved_distance:.2f}")
    print(f"Improvement over Nearest Neighbor: {nn_distance - improved_distance:.2f}")
    tsp.visualize_route(improved_route, "2-opt Improvement", 
                        save_path=os.path.join(vis_dir, "two_opt.png"))
    
    # Tavlama benzetimi
    print("Running Simulated Annealing algorithm...")
    sa_route, sa_history, sa_distance = tsp.simulated_annealing(save_dir=vis_dir)
    results["Simulated Annealing"] = {"distance": sa_distance, "route": sa_route, "improvement": nn_distance - sa_distance}
    print(f"Simulated Annealing distance: {sa_distance:.2f}")
    print(f"Improvement over Nearest Neighbor: {nn_distance - sa_distance:.2f}")
    tsp.visualize_route(sa_route, "Simulated Annealing", 
                        save_path=os.path.join(vis_dir, "simulated_annealing.png"))
    
    # Optimizasyon sürecini animasyonla göster
    tsp.animate_optimization(sa_history, 
                            save_path=os.path.join(vis_dir, "optimization_animation.gif"))
    
    # Save results to a file
    with open(os.path.join(vis_dir, "algorithm_results.txt"), "w") as f:
        f.write(f"TSP Results for {num_cities} cities\n")
        f.write("="*40 + "\n\n")
        
        # First, write a summary table
        f.write("Summary:\n")
        f.write(f"{'Algorithm':<20} {'Distance':<15} {'Improvement':<15}\n")
        f.write("-"*50 + "\n")
        
        for algorithm, data in results.items():
            improvement = data.get("improvement", "N/A")
            if isinstance(improvement, float):
                improvement_str = f"{improvement:.2f}"
            else:
                improvement_str = improvement
                
            f.write(f"{algorithm:<20} {data['distance']:<15.2f} {improvement_str:<15}\n")
        
        f.write("\n\n")
        
        # Then, write detailed information for each algorithm
        for algorithm, data in results.items():
            f.write(f"{algorithm}:\n")
            f.write(f"  Distance: {data['distance']:.2f}\n")
            if "improvement" in data:
                f.write(f"  Improvement over Nearest Neighbor: {data['improvement']:.2f}\n")
            f.write(f"  Route: {data['route']}\n\n")
    
    print(f"Results saved to {os.path.join(vis_dir, 'algorithm_results.txt')}")
    
    # Create a comparison image of all algorithms
    plt.figure(figsize=(15, 10))
    
    # Create 2x2 grid of subplots
    plt.subplot(2, 2, 1)
    
    # Plot Nearest Neighbor
    plt.scatter(tsp.cities[:, 0], tsp.cities[:, 1], c='red', s=100)
    for i, (x, y) in enumerate(tsp.cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    for i in range(len(nn_route)):
        j = (i + 1) % len(nn_route)
        plt.plot([tsp.cities[nn_route[i], 0], tsp.cities[nn_route[j], 0]],
                 [tsp.cities[nn_route[i], 1], tsp.cities[nn_route[j], 1]], 'b-')
    plt.scatter(tsp.cities[nn_route[0], 0], tsp.cities[nn_route[0], 1], c='green', s=150, zorder=3)
    plt.title(f"Nearest Neighbor: {nn_distance:.2f}")
    plt.grid(True)
    
    # Plot 2-opt
    plt.subplot(2, 2, 2)
    plt.scatter(tsp.cities[:, 0], tsp.cities[:, 1], c='red', s=100)
    for i, (x, y) in enumerate(tsp.cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    for i in range(len(improved_route)):
        j = (i + 1) % len(improved_route)
        plt.plot([tsp.cities[improved_route[i], 0], tsp.cities[improved_route[j], 0]],
                 [tsp.cities[improved_route[i], 1], tsp.cities[improved_route[j], 1]], 'b-')
    plt.scatter(tsp.cities[improved_route[0], 0], tsp.cities[improved_route[0], 1], c='green', s=150, zorder=3)
    plt.title(f"2-opt: {improved_distance:.2f}")
    plt.grid(True)
    
    # Plot Simulated Annealing
    plt.subplot(2, 2, 3)
    plt.scatter(tsp.cities[:, 0], tsp.cities[:, 1], c='red', s=100)
    for i, (x, y) in enumerate(tsp.cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    for i in range(len(sa_route)):
        j = (i + 1) % len(sa_route)
        plt.plot([tsp.cities[sa_route[i], 0], tsp.cities[sa_route[j], 0]],
                 [tsp.cities[sa_route[i], 1], tsp.cities[sa_route[j], 1]], 'b-')
    plt.scatter(tsp.cities[sa_route[0], 0], tsp.cities[sa_route[0], 1], c='green', s=150, zorder=3)
    plt.title(f"Simulated Annealing: {sa_distance:.2f}")
    plt.grid(True)
    
    # Leave fourth subplot empty or use it for something else
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, "Gezgin Satıcı Problemi\nKarşılaştırma", 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "algorithm_comparison.png"), dpi=300)
    plt.close()
