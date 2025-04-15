#bu çalışmada ackley fonksiyonunun iki boyutlu yukarıdan gösterimi kullanılarak, popülasyon temelli (TLBO) ve tekil arama yapan (Tabu Arama) iki algoritmanın kıyaslaması gösterilecek.
#TLBO için popülasyon sayısı 5 ve 10 olarak seçilecek ve iki görsel verileecek. 
#Her iki algoritma için de popülasyondaki bireylerin iyileşmesi çizgiler halinde gösterilerek tekil veya popülasyondaki bireyin optimuma yaklaşması görselleştirilmiş olacak.
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import random
import imageio




# Klasör oluşturma
# Python dosyasının bulunduğu klasörü al
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "visualizations")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ackley fonksiyonu tanımı
def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20

# Ackley fonksiyonunun görselleştirilmesi
def plot_ackley():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = ackley(X, Y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Ackley Fonksiyonu')
    
    return fig, ax, X, Y, Z

# TLBO (Teaching-Learning Based Optimization) algoritması
class TLBO:
    def __init__(self, pop_size, max_iter, bounds=(-5, 5)):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        
    def initialize(self):
        # Popülasyonu rastgele başlat
        self.population = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            size=(self.pop_size, 2)
        )
        self.fitness = np.zeros(self.pop_size)
        self.evaluate()
        self.history.append(self.population.copy())
        
    def evaluate(self):
        for i in range(self.pop_size):
            self.fitness[i] = ackley(self.population[i, 0], self.population[i, 1])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
                
    def teacher_phase(self):
        # En iyi çözüm öğretmen olarak seçilir
        teacher_index = np.argmin(self.fitness)
        teacher = self.population[teacher_index].copy()
        mean = np.mean(self.population, axis=0)
        
        # Öğretme faktörü
        tf = np.random.randint(1, 3, size=self.pop_size)
        
        # Yeni çözümler oluştur
        for i in range(self.pop_size):
            r = np.random.random(2)
            new_solution = self.population[i] + r * (teacher - tf[i] * mean)
            
            # Sınırları kontrol et
            new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
            
            # Yeni çözümü değerlendir
            new_fitness = ackley(new_solution[0], new_solution[1])
            
            # Daha iyi ise güncelle
            if new_fitness < self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution.copy()
    
    def learner_phase(self):
        for i in range(self.pop_size):
            # Rastgele başka bir öğrenci seç
            j = i
            while j == i:
                j = np.random.randint(0, self.pop_size)
            
            # Öğrenme
            if self.fitness[i] < self.fitness[j]:  # i daha iyi ise
                r = np.random.random(2)
                new_solution = self.population[i] + r * (self.population[i] - self.population[j])
            else:  # j daha iyi ise
                r = np.random.random(2)
                new_solution = self.population[i] + r * (self.population[j] - self.population[i])
            
            # Sınırları kontrol et
            new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
            
            # Yeni çözümü değerlendir
            new_fitness = ackley(new_solution[0], new_solution[1])
            
            # Daha iyi ise güncelle
            if new_fitness < self.fitness[i]:
                self.population[i] = new_solution
                self.fitness[i] = new_fitness
                
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution.copy()
    
    def optimize(self):
        self.initialize()
        
        for iteration in range(self.max_iter):
            self.teacher_phase()
            self.learner_phase()
            self.history.append(self.population.copy())
            
        return self.best_solution, self.best_fitness

# Tabu Arama algoritması
class TabuSearch:
    def __init__(self, max_iter=300, tabu_size=5, step_size=0.1, bounds=(-5, 5), diversification=1):
        self.max_iter = max_iter
        self.tabu_size = tabu_size
        self.step_size = step_size
        self.bounds = bounds
        self.tabu_list = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        self.diversification = diversification  # Farklı başlangıç noktaları sayısı
        
    def is_tabu(self, solution):
        for tabu_solution in self.tabu_list:
            if np.linalg.norm(solution - tabu_solution) < 0.1:
                return True
        return False
    
    def get_neighbors(self, solution):
        neighbors = []
        # Temel hareketler
        base_moves = [
            [self.step_size, 0],
            [-self.step_size, 0],
            [0, self.step_size],
            [0, -self.step_size],
            [self.step_size, self.step_size],
            [-self.step_size, -self.step_size],
            [self.step_size, -self.step_size],
            [-self.step_size, self.step_size]
        ]
        
        # Farklı ölçeklerde hareketler ekle
        scales = [1.0, 2.0, 0.5]
        
        for scale in scales:
            for move in base_moves:
                scaled_move = [m * scale for m in move]
                neighbor = solution + np.array(scaled_move)
                # Sınırları kontrol et
                neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
                neighbors.append(neighbor)
        
        # Tamamen rastgele komşular da ekle (çeşitliliği artırmak için)
        for _ in range(5):
            random_move = np.random.uniform(-self.step_size*2, self.step_size*2, size=2)
            neighbor = solution + random_move
            neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
            neighbors.append(neighbor)
            
        return neighbors
    
    def optimize(self, initial_solution=None):
        overall_best_solution = None
        overall_best_fitness = float('inf')
        overall_history = []
        
        # En iyi başlangıç noktasını bulmak için grid arama yapma
        grid_points = []
        grid_size = int(np.sqrt(self.diversification))
        grid_values = np.linspace(self.bounds[0], self.bounds[1], grid_size)
        
        for x in grid_values:
            for y in grid_values:
                grid_points.append(np.array([x, y]))
        
        # Kalan noktaları rastgele oluştur
        for _ in range(self.diversification - len(grid_points)):
            grid_points.append(np.random.uniform(self.bounds[0], self.bounds[1], size=2))
        
        # Her başlangıç noktası için optimizasyon yap
        for start_point in grid_points:
            current_solution = start_point.copy()
            current_fitness = ackley(current_solution[0], current_solution[1])
            local_best_solution = current_solution.copy()
            local_best_fitness = current_fitness
            
            # Tabu listesini temizle
            self.tabu_list = []
            local_history = [current_solution.copy()]
            
            # Yoğunlaştırma (intensification) aşaması
            iterations_without_improvement = 0
            max_iterations_without_improvement = self.max_iter // (2 * self.diversification)
            
            for iteration in range(self.max_iter // self.diversification):
                # Komşuları oluştur
                neighbors = self.get_neighbors(current_solution)
                
                # En iyi komşuyu bul
                best_neighbor = None
                best_neighbor_fitness = float('inf')
                
                for neighbor in neighbors:
                    if not self.is_tabu(neighbor):
                        fitness = ackley(neighbor[0], neighbor[1])
                        if fitness < best_neighbor_fitness:
                            best_neighbor = neighbor
                            best_neighbor_fitness = fitness
                
                # Eğer tüm komşular tabu ise veya çok uzun zamandır iyileşme yoksa
                # aspiration kriterini esnet
                if best_neighbor is None or iterations_without_improvement > max_iterations_without_improvement:
                    for neighbor in neighbors:
                        fitness = ackley(neighbor[0], neighbor[1])
                        if fitness < best_neighbor_fitness:
                            best_neighbor = neighbor
                            best_neighbor_fitness = fitness
                
                # Tabu listesini güncelle
                self.tabu_list.append(current_solution.copy())
                if len(self.tabu_list) > self.tabu_size:
                    self.tabu_list.pop(0)
                
                # Çözümü güncelle
                current_solution = best_neighbor.copy()
                current_fitness = best_neighbor_fitness
                
                # Global en iyi çözümü güncelle
                if current_fitness < local_best_fitness:
                    local_best_solution = current_solution.copy()
                    local_best_fitness = current_fitness
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
                
                # Diversifikasyon: Eğer uzun süre iyileşme yoksa rastgele bir noktaya atla
                if iterations_without_improvement > max_iterations_without_improvement:
                    # Yeni bir rastgele nokta oluştur, ancak global optimuma biraz daha yakın
                    current_solution = np.random.uniform(-2, 2, size=2)
                    current_fitness = ackley(current_solution[0], current_solution[1])
                    iterations_without_improvement = 0
                
                local_history.append(current_solution.copy())
            
            # Bu başlangıç noktasından bulunan en iyi çözümü kontrol et
            if local_best_fitness < overall_best_fitness:
                overall_best_solution = local_best_solution.copy()
                overall_best_fitness = local_best_fitness
                overall_history = local_history.copy()
        
        self.best_solution = overall_best_solution
        self.best_fitness = overall_best_fitness
        self.history = overall_history
            
        return self.best_solution, self.best_fitness

# Algoritmaları çalıştır ve görselleştir
def run_and_visualize():
    # Ackley fonksiyonunu görselleştir
    fig, ax, X, Y, Z = plot_ackley()
    
    # TLBO (5 popülasyon)
    print("TLBO (5 popülasyon) çalıştırılıyor...")
    tlbo_5 = TLBO(pop_size=5, max_iter=60)
    tlbo_5_best, tlbo_5_fitness = tlbo_5.optimize()
    
    # TLBO (10 popülasyon)
    print("TLBO (10 popülasyon) çalıştırılıyor...")
    tlbo_10 = TLBO(pop_size=10, max_iter=30)
    tlbo_10_best, tlbo_10_fitness = tlbo_10.optimize()
    
    # Tabu Arama
    print("Tabu Arama çalıştırılıyor...")
    tabu = TabuSearch(max_iter=1500, tabu_size=25, step_size=0.5, diversification=9)
    tabu_best, tabu_fitness = tabu.optimize()
    
    # Sonuçları yazdır
    print("\n=== Optimizasyon Sonuçları ===")
    print(f"TLBO (5 popülasyon) en iyi çözüm: {tlbo_5_best}, fitness: {tlbo_5_fitness}")
    print(f"TLBO (10 popülasyon) en iyi çözüm: {tlbo_10_best}, fitness: {tlbo_10_fitness}")
    print(f"Tabu Arama en iyi çözüm: {tabu_best}, fitness: {tabu_fitness}")
    print("===============================")
    
    # TLBO (5 popülasyon) animasyonu
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    contour = ax1.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('TLBO (5 popülasyon) Optimizasyon Süreci')
    
    # Başlangıç popülasyonunu çiz
    scatter = ax1.scatter(tlbo_5.history[0][:, 0], tlbo_5.history[0][:, 1], c='red', s=50)
    
    # Animasyon için fonksiyon
    def update_tlbo_5(frame):
        ax1.clear()
        contour = ax1.contourf(X, Y, Z, 50, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'TLBO (5 popülasyon) - İterasyon {frame}')
        
        # Önceki iterasyonlardaki yolları çiz
        for i in range(tlbo_5.pop_size):
            path_x = [tlbo_5.history[j][i, 0] for j in range(min(frame+1, len(tlbo_5.history)))]
            path_y = [tlbo_5.history[j][i, 1] for j in range(min(frame+1, len(tlbo_5.history)))]
            ax1.plot(path_x, path_y, 'b-', alpha=0.5)
        
        # Mevcut popülasyonu çiz
        if frame < len(tlbo_5.history):
            ax1.scatter(tlbo_5.history[frame][:, 0], tlbo_5.history[frame][:, 1], c='red', s=50)
        
        # Global optimum noktasını işaretle
        ax1.scatter(0, 0, c='green', s=100, marker='*', label='Global Optimum')
        ax1.legend()
        
        return contour,
    
    # Animasyonu oluştur ve kaydet
    ani_tlbo_5 = FuncAnimation(fig1, update_tlbo_5, frames=len(tlbo_5.history), interval=200, blit=False)
    ani_tlbo_5.save(os.path.join(output_dir, 'tlbo_5_animation.gif'), writer='pillow', fps=5)
    
    # TLBO (10 popülasyon) animasyonu
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    contour = ax2.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('TLBO (10 popülasyon) Optimizasyon Süreci')
    
    # Başlangıç popülasyonunu çiz
    scatter = ax2.scatter(tlbo_10.history[0][:, 0], tlbo_10.history[0][:, 1], c='red', s=50)
    
    # Animasyon için fonksiyon
    def update_tlbo_10(frame):
        ax2.clear()
        contour = ax2.contourf(X, Y, Z, 50, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'TLBO (10 popülasyon) - İterasyon {frame}')
        
        # Önceki iterasyonlardaki yolları çiz
        for i in range(tlbo_10.pop_size):
            path_x = [tlbo_10.history[j][i, 0] for j in range(min(frame+1, len(tlbo_10.history)))]
            path_y = [tlbo_10.history[j][i, 1] for j in range(min(frame+1, len(tlbo_10.history)))]
            ax2.plot(path_x, path_y, 'b-', alpha=0.5)
        
        # Mevcut popülasyonu çiz
        if frame < len(tlbo_10.history):
            ax2.scatter(tlbo_10.history[frame][:, 0], tlbo_10.history[frame][:, 1], c='red', s=50)
        
        # Global optimum noktasını işaretle
        ax2.scatter(0, 0, c='green', s=100, marker='*', label='Global Optimum')
        ax2.legend()
        
        return contour,
    
    # Animasyonu oluştur ve kaydet
    ani_tlbo_10 = FuncAnimation(fig2, update_tlbo_10, frames=len(tlbo_10.history), interval=200, blit=False)
    ani_tlbo_10.save(os.path.join(output_dir, 'tlbo_10_animation.gif'), writer='pillow', fps=5)
    
    # Tabu Arama animasyonu
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    contour = ax3.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Tabu Arama Optimizasyon Süreci')
    
    # Başlangıç noktasını çiz
    scatter = ax3.scatter(tabu.history[0][0], tabu.history[0][1], c='red', s=50)
    
    # Animasyon için fonksiyon
    def update_tabu(frame):
        ax3.clear()
        contour = ax3.contourf(X, Y, Z, 50, cmap='viridis')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title(f'Tabu Arama - İterasyon {frame}')
        
        # Önceki iterasyonlardaki yolu çiz
        path_x = [tabu.history[j][0] for j in range(min(frame+1, len(tabu.history)))]
        path_y = [tabu.history[j][1] for j in range(min(frame+1, len(tabu.history)))]
        ax3.plot(path_x, path_y, 'r-', alpha=0.7)
        
        # Mevcut noktayı çiz
        if frame < len(tabu.history):
            ax3.scatter(tabu.history[frame][0], tabu.history[frame][1], c='red', s=50)
        
        # Tabu listesini çiz
        if frame > 0:
            tabu_list_at_frame = tabu.tabu_list[:min(frame, tabu.tabu_size)]
            tabu_x = [sol[0] for sol in tabu_list_at_frame]
            tabu_y = [sol[1] for sol in tabu_list_at_frame]
            ax3.scatter(tabu_x, tabu_y, c='blue', s=30, alpha=0.5, marker='x', label='Tabu Listesi')
        
        # Global optimum noktasını işaretle
        ax3.scatter(0, 0, c='green', s=100, marker='*', label='Global Optimum')
        ax3.legend()
        
        return contour,
    
    # Animasyonu oluştur ve kaydet
    ani_tabu = FuncAnimation(fig3, update_tabu, frames=len(tabu.history), interval=200, blit=False)
    ani_tabu.save(os.path.join(output_dir, 'tabu_animation.gif'), writer='pillow', fps=5)
    
    # Karşılaştırma grafiği
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    contour = ax4.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax4)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Algoritma Karşılaştırması')
    
    # TLBO (5 popülasyon) yollarını çiz
    for i in range(tlbo_5.pop_size):
        path_x = [tlbo_5.history[j][i, 0] for j in range(len(tlbo_5.history))]
        path_y = [tlbo_5.history[j][i, 1] for j in range(len(tlbo_5.history))]
        ax4.plot(path_x, path_y, 'b-', alpha=0.3)
    
    # TLBO (10 popülasyon) yollarını çiz
    for i in range(tlbo_10.pop_size):
        path_x = [tlbo_10.history[j][i, 0] for j in range(len(tlbo_10.history))]
        path_y = [tlbo_10.history[j][i, 1] for j in range(len(tlbo_10.history))]
        ax4.plot(path_x, path_y, 'g-', alpha=0.3)
    
    # Tabu Arama yolunu çiz
    path_x = [tabu.history[j][0] for j in range(len(tabu.history))]
    path_y = [tabu.history[j][1] for j in range(len(tabu.history))]
    ax4.plot(path_x, path_y, 'r-', alpha=0.7)
    
    # Son popülasyonları çiz
    ax4.scatter(tlbo_5.history[-1][:, 0], tlbo_5.history[-1][:, 1], c='blue', s=50, label='TLBO (5)')
    ax4.scatter(tlbo_10.history[-1][:, 0], tlbo_10.history[-1][:, 1], c='green', s=50, label='TLBO (10)')
    ax4.scatter(tabu.history[-1][0], tabu.history[-1][1], c='red', s=50, label='Tabu Arama')
    
    # Global optimum noktasını işaretle
    ax4.scatter(0, 0, c='black', s=100, marker='*', label='Global Optimum')
    ax4.legend()
    
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'))
    
    plt.close('all')

if __name__ == "__main__":
    run_and_visualize()
