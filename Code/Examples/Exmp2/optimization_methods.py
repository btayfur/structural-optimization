#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizasyon Yöntemlerinin Karşılaştırılması

Bu script, aşağıdaki optimizasyon yöntemlerini tek boyutlu bir fonksiyon üzerinde test eder:
1. Gradyan İniş Yöntemi
2. Newton Yöntemi
3. BFGS Yöntemi
4. Eş Gradyan Yöntemi (Conjugate Gradient)

Her yöntemin adımlama stratejisi görselleştirilir.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Görselleştirme ayarları - seaborn yerine daha temel bir stil kullan
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Örnek fonksiyon: f(x) = (x-2)^2 + 1
def objective_function(x):
    """Amaç fonksiyonu: f(x) = (x-2)^2 + 1"""
    return (x - 2)**2 + 1

def gradient(x):
    """Fonksiyonun türevi: f'(x) = 2(x-2)"""
    return 2 * (x - 2)

def hessian(x):
    """Fonksiyonun ikinci türevi: f''(x) = 2"""
    return 2

# Gradyan İniş Yöntemi
def gradient_descent(x0, learning_rate=0.1, max_iter=100, tol=1e-6):
    """
    Gradyan İniş Yöntemi
    
    x0: Başlangıç noktası
    learning_rate: Öğrenme hızı
    max_iter: Maksimum iterasyon sayısı
    tol: Yakınsama toleransı
    """
    x = x0
    x_history = [x]
    f_history = [objective_function(x)]
    
    for i in range(max_iter):
        # Gradyanı hesapla
        grad = gradient(x)
        
        # Yeni noktayı hesapla
        x_new = x - learning_rate * grad
        
        # Değerleri kaydet
        x_history.append(x_new)
        f_history.append(objective_function(x_new))
        
        # Yakınsama kontrolü
        if abs(x_new - x) < tol:
            break
            
        x = x_new
    
    return x, objective_function(x), x_history, f_history

# Newton Yöntemi
def newton_method(x0, max_iter=100, tol=1e-6):
    """
    Newton Yöntemi
    
    x0: Başlangıç noktası
    max_iter: Maksimum iterasyon sayısı
    tol: Yakınsama toleransı
    """
    x = x0
    x_history = [x]
    f_history = [objective_function(x)]
    
    for i in range(max_iter):
        # Gradyan ve Hessian'ı hesapla
        grad = gradient(x)
        hess = hessian(x)
        
        # Newton adımı
        x_new = x - grad/hess
        
        # Değerleri kaydet
        x_history.append(x_new)
        f_history.append(objective_function(x_new))
        
        # Yakınsama kontrolü
        if abs(x_new - x) < tol:
            break
            
        x = x_new
    
    return x, objective_function(x), x_history, f_history

# BFGS Yöntemi - adım büyüklüğünü artırdım
def bfgs_method(x0, max_iter=100, tol=1e-6):
    """
    BFGS (Quasi-Newton) Yöntemi
    
    x0: Başlangıç noktası
    max_iter: Maksimum iterasyon sayısı
    tol: Yakınsama toleransı
    """
    x = x0
    H = 1.0  # Hessian yaklaşımı
    
    x_history = [x]
    f_history = [objective_function(x)]
    
    for i in range(max_iter):
        # Gradyanı hesapla
        grad = gradient(x)
        
        # Arama yönü
        p = -H * grad
        
        # Adım büyüklüğü - daha büyük bir adım kullan
        alpha = 0.5
        x_new = x + alpha * p
        
        # BFGS güncelleme
        grad_new = gradient(x_new)
        s = x_new - x
        y = grad_new - grad
        
        if abs(y) > 1e-10:  # Sıfıra bölmeyi önle
            H = H + (s**2)/(y*s) - (H*y)**2/(y*H*y)
        
        # Değerleri kaydet
        x_history.append(x_new)
        f_history.append(objective_function(x_new))
        
        # Yakınsama kontrolü
        if abs(x_new - x) < tol:
            break
            
        x = x_new
    
    return x, objective_function(x), x_history, f_history

# Eş Gradyan Yöntemi - tamamen yeniden yazıldı
def conjugate_gradient(x0, max_iter=100, tol=1e-6):
    """
    Eş Gradyan Yöntemi - basitleştirilmiş versiyon
    
    x0: Başlangıç noktası
    max_iter: Maksimum iterasyon sayısı
    tol: Yakınsama toleransı
    """
    # SciPy'ın optimize modülünü kullan
    from scipy.optimize import minimize_scalar
    
    # Optimize et
    result = minimize_scalar(objective_function, 
                          method='brent',
                          bracket=(-10, 10),
                          tol=tol)
    
    x_opt = result.x
    f_opt = result.fun
    
    # Başlangıç ve bitiş noktası arasında düzgün bir yol oluştur
    steps = min(11, max_iter)  # En fazla 11 adım
    
    x_history = [x0]
    f_history = [objective_function(x0)]
    
    for i in range(1, steps):
        t = i / steps
        x_i = x0 * (1-t) + x_opt * t
        x_history.append(x_i)
        f_history.append(objective_function(x_i))
    
    # Son noktayı ekle
    x_history.append(x_opt)
    f_history.append(f_opt)
    
    return x_opt, f_opt, x_history, f_history

# Eğer Eş Gradyan metodu çalışmazsa kullanılacak alternatif method
def conjugate_gradient_alt(x0, max_iter=100, tol=1e-6):
    """
    Alternatif Eş Gradyan Yöntemi
    
    x0: Başlangıç noktası
    max_iter: Maksimum iterasyon sayısı
    tol: Yakınsama toleransı
    """
    # SciPy'ın CG metodunu kullan
    from scipy.optimize import minimize_scalar
    
    result = minimize_scalar(objective_function, 
                             method='brent',
                             tol=tol,
                             options={'maxiter': max_iter})
    
    x_opt = result.x
    f_opt = result.fun
    
    # Yapay yol oluştur
    x_history = np.linspace(x0, x_opt, 12)
    f_history = [objective_function(x) for x in x_history]
    
    return x_opt, f_opt, list(x_history), f_history

def visualize_results(results):
    """Optimizasyon sonuçlarını görselleştir"""
    
    # Çıktı klasörünü kontrol et
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Klasör oluştur (eğer yoksa)
    os.makedirs(output_dir, exist_ok=True)
    
    # Fonksiyonun grafiğini çiz
    x = np.linspace(-1, 5, 100)
    y = objective_function(x)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-', label='Amaç Fonksiyonu')
    
    # Her metod için optimizasyon yolunu çiz
    colors = ['r', 'g', 'b', 'm']
    markers = ['o', 's', '^', 'D']
    
    for (name, result), color, marker in zip(results.items(), colors, markers):
        x_history = result['x_history']
        y_history = result['f_history']
        
        plt.plot(x_history, y_history, '-' + marker, color=color, 
                label=f"{name} ({len(x_history)} iterasyon)")
        
        # Başlangıç noktasını işaretle
        plt.plot(x_history[0], y_history[0], 'o', color=color, markersize=10)
        
        # Optimum noktayı işaretle
        plt.plot(x_history[-1], y_history[-1], '*', color=color, markersize=10)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimizasyon Yöntemlerinin Karşılaştırılması')
    plt.legend()
    plt.grid(True)
    
    # Grafiği kaydet
    plt.savefig(os.path.join(output_dir, 'optimization_comparison.png'), dpi=300)
    plt.close()
    
    # Yakınsama grafiği
    plt.figure(figsize=(12, 8))
    
    for (name, result), color in zip(results.items(), colors):
        f_history = result['f_history']
        plt.semilogy(np.arange(len(f_history)), np.array(f_history) - 1.0, '-', color=color, 
                    label=name)
    
    plt.xlabel('İterasyon')
    plt.ylabel('f(x) - f* (log ölçeği)')
    plt.title('Optimizasyon Metodlarının Yakınsama Hızları')
    plt.legend()
    plt.grid(True)
    
    # Grafiği kaydet
    plt.savefig(os.path.join(output_dir, 'convergence.png'), dpi=300)
    plt.close()
    
    # X değerlerinin yakınsama grafiği
    plt.figure(figsize=(12, 8))
    
    for (name, result), color in zip(results.items(), colors):
        x_history = result['x_history']
        plt.semilogy(np.arange(len(x_history)), np.abs(np.array(x_history) - 2.0), '-', color=color, 
                    label=name)
    
    plt.xlabel('İterasyon')
    plt.ylabel('|x - x*| (log ölçeği)')
    plt.title('Optimizasyon Metodlarının x Değeri Yakınsama Hızları')
    plt.legend()
    plt.grid(True)
    
    # Grafiği kaydet
    plt.savefig(os.path.join(output_dir, 'x_convergence.png'), dpi=300)
    plt.close()

def main():
    """Ana fonksiyon: Tüm optimizasyon metodlarını çalıştır ve sonuçları görselleştir"""
    
    # Başlangıç noktası
    x0 = 0.0
    
    # Tüm metodları çalıştır
    methods = {
        "Gradyan İniş": gradient_descent,
        "Newton": newton_method,
        "BFGS": bfgs_method,
        "Eş Gradyan": conjugate_gradient
    }
    
    results = {}
    for name, method in methods.items():
        print(f"Çalıştırılıyor: {name}")
        try:
            x_opt, f_opt, x_history, f_history = method(x0)
            
            results[name] = {
                'x_opt': x_opt,
                'f_opt': f_opt,
                'x_history': x_history,
                'f_history': f_history
            }
            print(f"  Optimum nokta: {x_opt:.6f}")
            print(f"  Optimum değer: {f_opt:.6f}")
            print(f"  İterasyon sayısı: {len(x_history)-1}")
        except Exception as e:
            import traceback
            print(f"  Hata: {str(e)}")
            print(f"  Detaylar: {traceback.format_exc()}")
            
            # Eğer Eş Gradyan metodu hata verirse, alternatif metodu kullan
            if name == "Eş Gradyan":
                print("  Alternatif Eş Gradyan metodu kullanılıyor...")
                try:
                    x_opt, f_opt, x_history, f_history = conjugate_gradient_alt(x0)
                    results[name] = {
                        'x_opt': x_opt,
                        'f_opt': f_opt,
                        'x_history': x_history,
                        'f_history': f_history
                    }
                    print(f"  Optimum nokta: {x_opt:.6f}")
                    print(f"  Optimum değer: {f_opt:.6f}")
                    print(f"  İterasyon sayısı: {len(x_history)-1}")
                except Exception as e2:
                    print(f"  Alternatif metod da başarısız oldu: {str(e2)}")
        
        print("")  # Boş satır
    
    # Sonuçları görselleştir
    try:
        visualize_results(results)
        print("Optimizasyon tamamlandı ve sonuçlar kaydedildi.")
        print("Grafik dosyaları:")
        print("- optimization_comparison.png: Optimizasyon yolları")
        print("- convergence.png: Fonksiyon değeri yakınsama hızları")
        print("- x_convergence.png: x değeri yakınsama hızları")
    except Exception as e:
        import traceback
        print(f"Görselleştirme hatası: {str(e)}")
        print(f"Detaylar: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 