import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import os
import matplotlib as mpl

# Görseller için klasör oluştur
# Dosyanın bulunduğu dizini al ve visualizations klasörünü orada oluştur
current_dir = os.path.dirname(os.path.abspath(__file__))
visualization_dir = os.path.join(current_dir, 'visualizations')
os.makedirs(visualization_dir, exist_ok=True)

# Akademik stil ayarları
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Georgia'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.5,
    'lines.markersize': 6
})

# Renk paleti (ColorBlind friendly)
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']

class AckleyOptimizer:
    def __init__(self, dimensions: int, bounds: Tuple[float, float] = (-32.768, 32.768)):
        self.dimensions = dimensions
        self.bounds = bounds
        
    def ackley_function(self, x: np.ndarray) -> float:
        """
        Ackley fonksiyonu implementasyonu
        f(x) = -20 * exp(-0.2 * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2π*x_i))) + 20 + e
        """
        first_sum = np.sum(x**2)
        second_sum = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(first_sum / self.dimensions))
        term2 = -np.exp(second_sum / self.dimensions)
        
        return term1 + term2 + 20 + np.e
    
    def simulated_annealing(self, 
                          initial_temp: float = 100.0,
                          min_temp: float = 1e-8,
                          cooling_rate: float = 0.99,
                          iterations_per_temp: int = 100) -> Dict:
        """
        İyileştirilmiş tavlama benzetimi algoritması
        """
        # Rastgele başlangıç noktası oluştur
        current_solution = np.random.uniform(self.bounds[0], 
                                          self.bounds[1], 
                                          self.dimensions)
        current_energy = self.ackley_function(current_solution)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temperature = initial_temp
        energy_history = [current_energy]
        distance_history = [np.linalg.norm(current_solution)]  # Orijinden uzaklık
        temperature_history = [temperature]
        step_size_history = []
        acceptance_ratio_history = []
        
        iteration = 0
        total_iterations = 0
        
        # Sıcaklık minimum değerin üzerinde olduğu sürece devam et
        while temperature > min_temp:
            accepted_moves = 0
            
            for _ in range(iterations_per_temp):
                # Yeni çözüm öner (sıcaklığa bağlı adım büyüklüğü)
                step_size = np.sqrt(temperature) * (self.bounds[1] - self.bounds[0]) / 10
                new_solution = current_solution + np.random.normal(0, step_size, self.dimensions)
                new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
                
                # Yeni çözümün enerjisini hesapla
                new_energy = self.ackley_function(new_solution)
                
                # Kabul olasılığını hesapla (numerik stabilite için min kullanıldı)
                delta_energy = new_energy - current_energy
                acceptance_probability = np.exp(-min(100, max(-100, delta_energy / temperature)))
                
                # Yeni çözümü kabul et veya reddet
                if delta_energy < 0 or np.random.random() < acceptance_probability:
                    current_solution = new_solution
                    current_energy = new_energy
                    accepted_moves += 1
                    
                    if current_energy < best_energy:
                        best_solution = current_solution.copy()
                        best_energy = current_energy
                
                energy_history.append(current_energy)
                distance_history.append(np.linalg.norm(current_solution))
                
                total_iterations += 1
                
            # Her sıcaklık adımı için metrikleri kaydet
            step_size_history.append(step_size)
            acceptance_ratio = accepted_moves / iterations_per_temp
            acceptance_ratio_history.append(acceptance_ratio)
            temperature_history.append(temperature)
            
            # Sıcaklığı düşür
            temperature *= cooling_rate
            iteration += 1
            
        return {
            'best_solution': best_solution,
            'best_energy': best_energy,
            'energy_history': energy_history,
            'distance_history': distance_history,
            'temperature_history': temperature_history,
            'step_size_history': step_size_history,
            'acceptance_ratio_history': acceptance_ratio_history,
            'total_iterations': total_iterations,
            'iterations': iteration
        }

def test_different_dimensions():
    """
    Farklı boyutlarda test fonksiyonu
    """
    dimensions = [2, 5, 10, 20]
    results = []
    
    for dim in dimensions:
        print(f"\nBoyut {dim} için optimizasyon başladı...")
        optimizer = AckleyOptimizer(dimensions=dim)
        start_time = time.time()
        result_dict = optimizer.simulated_annealing()
        end_time = time.time()
        
        result_dict['dimension'] = dim
        result_dict['execution_time'] = end_time - start_time
        results.append(result_dict)
        
        # Her boyut için ayrı görselleştirme yap
        visualize_single_dimension(result_dict)
        
    return results

def set_plot_style(ax, title, xlabel, ylabel):
    """
    Grafik stilini ayarlamak için yardımcı fonksiyon
    """
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax

def visualize_single_dimension(result: Dict):
    """
    Tek bir boyut için detaylı görselleştirme
    """
    dim = result['dimension']
    
    # 1. Yakınsama grafiği
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result['energy_history'], color=COLORS[0], linewidth=1.5)
    ax.set_yscale('log')  # Logaritmik ölçek
    
    # Yakınsamayı gösteren yeşil nokta
    min_energy_idx = np.argmin(result['energy_history'])
    ax.scatter(min_energy_idx, result['energy_history'][min_energy_idx], 
              color='red', s=80, zorder=5, label=f'En iyi: {result["best_energy"]:.6f}')
    
    set_plot_style(ax, f'Boyut {dim} için Yakınsama Analizi', 'İterasyon Sayısı', 'Enerji Değeri (log ölçek)')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Bilgilendirici metin ekle
    ax.text(0.02, 0.02, f'En düşük enerji: {result["best_energy"]:.6f}\n'
                      f'İterasyon: {min_energy_idx}', 
          transform=ax.transAxes, 
          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
          fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(visualization_dir, f'convergence_dim_{dim}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 2. Adım büyüklüğü ve kabul oranı değişimi
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color1 = COLORS[0]
    ax1.set_xlabel('Sıcaklık Adımı')
    ax1.set_ylabel('Adım Büyüklüğü', color=color1)
    line1 = ax1.plot(range(len(result['step_size_history'])), result['step_size_history'], 
                    'o-', color=color1, label='Adım Büyüklüğü')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = COLORS[1]
    ax2.set_ylabel('Kabul Oranı', color=color2)
    line2 = ax2.plot(range(len(result['acceptance_ratio_history'])), result['acceptance_ratio_history'], 
                    'o-', color=color2, label='Kabul Oranı')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Birleştirilmiş başlık ve diğer elementler
    ax1.set_title(f'Boyut {dim} için Adım Büyüklüğü ve Kabul Oranı Değişimi', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    # Birleştirilmiş legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', frameon=True, fancybox=True, shadow=True)
    
    fig.tight_layout()
    save_path = os.path.join(visualization_dir, f'step_size_acceptance_dim_{dim}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 3. Sıcaklık ve enerji değişimi
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color1 = COLORS[2]
    ax1.set_xlabel('Sıcaklık Adımı')
    ax1.set_ylabel('Sıcaklık (log ölçek)', color=color1)
    line1 = ax1.plot(range(len(result['temperature_history'])), result['temperature_history'], 
                    'o-', color=color1, label='Sıcaklık')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Enerji değerlerini örnekle
    energy_samples = result['energy_history'][::len(result['energy_history'])//len(result['temperature_history'])]
    if len(energy_samples) > len(result['temperature_history']):
        energy_samples = energy_samples[:len(result['temperature_history'])]
    elif len(energy_samples) < len(result['temperature_history']):
        energy_samples = np.pad(energy_samples, (0, len(result['temperature_history']) - len(energy_samples)), 'edge')
    
    ax2 = ax1.twinx()
    color2 = COLORS[3]
    ax2.set_ylabel('Enerji (log ölçek)', color=color2)
    line2 = ax2.plot(range(len(energy_samples)), energy_samples, 
                    'o-', color=color2, label='Enerji')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Birleştirilmiş başlık ve diğer elementler
    ax1.set_title(f'Boyut {dim} için Sıcaklık ve Enerji Değişimi', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Birleştirilmiş legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', frameon=True, fancybox=True, shadow=True)
    
    fig.tight_layout()
    save_path = os.path.join(visualization_dir, f'temperature_energy_dim_{dim}.png')
    plt.savefig(save_path)
    plt.close()
    
    # 4. Çözüm yolu (sadece 2 boyut için)
    if dim == 2:
        visualize_solution_path_2d(result)

def visualize_solution_path_2d(result: Dict):
    """
    2 boyutlu uzayda çözüm yolunu görselleştir
    """
    # x ve y için mesh grid oluştur
    x = np.linspace(-5, 5, 200)  # Daha yüksek çözünürlük
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    
    # Mesh grid üzerinde Ackley fonksiyonunu hesapla
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            optimizer = AckleyOptimizer(dimensions=2)
            Z[j, i] = optimizer.ackley_function(np.array([X[j, i], Y[j, i]]))
    
    # Çözüm yolunu oluştur
    energy_history = np.array(result['energy_history'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Kontur çizimi - viridis yerine daha akademik bir renk paleti kullan
    contour = ax.contourf(X, Y, Z, 50, cmap='plasma')
    contour_lines = ax.contour(X, Y, Z, 10, colors='white', linewidths=0.5, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    cbar = plt.colorbar(contour, ax=ax, pad=0.01)
    cbar.ax.set_ylabel('Ackley Fonksiyonu Değeri', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # Başlangıç ve bitiş noktaları
    best_solution = result['best_solution']
    ax.plot(0, 0, 'w*', markersize=18, label='Global Minimum (0,0)', 
          markeredgecolor='black', markeredgewidth=1)
    ax.plot(best_solution[0], best_solution[1], 'ro', markersize=12, 
          label=f'En İyi Çözüm ({best_solution[0]:.4f}, {best_solution[1]:.4f})',
          markeredgecolor='white', markeredgewidth=1)
    
    # Eksen etiketleri ve başlık
    ax.set_title('Ackley Fonksiyonu - 2D Optimizasyon Sonucu', fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    # Izgara çizgilerini daha akademik yap
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Çerçeveyi görünür hale getir
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    
    # Efsane kutusu daha akademik olsun
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    
    # Bilgilendirici metin ekle
    ax.text(0.02, 0.02, f'En düşük enerji: {result["best_energy"]:.6f}\n'
                      f'Konum: ({best_solution[0]:.6f}, {best_solution[1]:.6f})',
          transform=ax.transAxes, 
          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), 
          fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(visualization_dir, 'ackley_contour_2d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_results(results: List[dict]):
    """
    Tüm boyutlar için karşılaştırmalı görselleştirme fonksiyonu
    """
    # Enerji değişim grafiği
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, result in enumerate(results):
        # Her 10 noktadan birini çiz (grafiği daha temiz göstermek için)
        history = result['energy_history'][::10]
        ax.plot(range(0, len(result['energy_history']), 10), history,
               color=COLORS[i % len(COLORS)], linewidth=1.8,
               label=f'Boyut {result["dimension"]}')
    
    set_plot_style(ax, 'Boyutlara Göre Enerji Değişimi', 'İterasyon (her 10 adımdan biri)', 'Enerji')
    ax.legend(frameon=True, fancybox=True, shadow=True, title='Problem Boyutu')
    
    # Logaritmik ölçekte koy (opsiyonel)
    # ax.set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(visualization_dir, 'energy_history_all_dims.png')
    plt.savefig(save_path)
    plt.close()
    
    # Boyut-Enerji ilişkisi
    dimensions = [r['dimension'] for r in results]
    best_energies = [r['best_energy'] for r in results]
    exec_times = [r['execution_time'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # İlk grafik: Boyut - En İyi Enerji
    ax1.plot(dimensions, best_energies, 'o-', color=COLORS[0], linewidth=2, markersize=8)
    set_plot_style(ax1, 'Boyut - En İyi Enerji İlişkisi', 'Problem Boyutu', 'En İyi Enerji Değeri')
    
    # Veri noktalarına değerleri ekle
    for i, (dim, energy) in enumerate(zip(dimensions, best_energies)):
        ax1.annotate(f'{energy:.6f}', xy=(dim, energy), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    # İkinci grafik: Boyut - Çalışma Süresi
    ax2.plot(dimensions, exec_times, 'o-', color=COLORS[1], linewidth=2, markersize=8)
    set_plot_style(ax2, 'Boyut - Çalışma Süresi İlişkisi', 'Problem Boyutu', 'Çalışma Süresi (s)')
    
    # Veri noktalarına değerleri ekle
    for i, (dim, time_val) in enumerate(zip(dimensions, exec_times)):
        ax2.annotate(f'{time_val:.2f}s', xy=(dim, time_val), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    # Genel düzen ayarlamaları
    plt.suptitle('Problem Boyutunun Performans Üzerindeki Etkisi', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect, suptitle için yer açar
    save_path = os.path.join(visualization_dir, 'dimension_analysis.png')
    plt.savefig(save_path)
    plt.close()
    
    # Boyut - Kabul Oranı İlişkisi
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, result in enumerate(results):
        ax.plot(range(len(result['acceptance_ratio_history'])), 
               result['acceptance_ratio_history'], 'o-',
               color=COLORS[i % len(COLORS)], linewidth=1.8,
               label=f'Boyut {result["dimension"]}')
    
    set_plot_style(ax, 'Boyutlara Göre Kabul Oranı Değişimi', 'Sıcaklık Adımı', 'Kabul Oranı')
    ax.legend(frameon=True, fancybox=True, shadow=True, title='Problem Boyutu')
    
    plt.tight_layout()
    save_path = os.path.join(visualization_dir, 'acceptance_ratio_all_dims.png')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Farklı boyutlarda test et
    results = test_different_dimensions()
    
    # Tüm boyutların karşılaştırmalı sonuçlarını görselleştir
    plot_results(results)
    
    # Sonuçları yazdır
    print("\nTest Sonuçları:")
    print("-" * 50)
    for result in results:
        print(f"Boyut: {result['dimension']}")
        print(f"En iyi enerji: {result['best_energy']:.6f}")
        print(f"En iyi çözüm: {result['best_solution']}")
        print(f"Toplam iterasyon sayısı: {result['total_iterations']}")
        print(f"Sıcaklık adımı sayısı: {result['iterations']}")
        print(f"Çalışma süresi: {result['execution_time']:.2f} saniye")
        print("-" * 50) 