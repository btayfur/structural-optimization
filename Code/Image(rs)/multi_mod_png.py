import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn minimalist tema ayarları
sns.set_theme(style="white", font="sans-serif", font_scale=1.3)

# Fonksiyon tanımı (Ackley 1D)
def ackley(x, a=20, b=0.2, c=2*np.pi):
    x = np.array(x)
    term1 = -a * np.exp(-b * np.abs(x))
    term2 = -np.exp(np.cos(c * x))
    return term1 + term2 + a + np.exp(1)

# Değerler
x = np.linspace(-10, 10, 1000)
y = ackley(x)

# Renk paleti (profesyonel, nötr)
main_color = "#1f2937"       # koyu gri-mavi (navy-gray)
accent_color = "#e11d48"     # soft crimson

# Grafik çizimi
fig, ax = plt.subplots(figsize=(12, 6))

# Fonksiyon eğrisi
ax.plot(x, y, color=main_color, linewidth=2.5, label="Ackley Function (1D)")

# Global minimum noktası
ax.axvline(x=0, color=accent_color, linestyle="--", linewidth=1.8)
ax.scatter(0, ackley(0), color=accent_color, s=80, zorder=5)

# Not ve etiket
ax.annotate("Global Minimum\nx = 0", xy=(0, ackley(0)), xytext=(1.5, 4),
            arrowprops=dict(arrowstyle="->", color=accent_color),
            fontsize=12, color=accent_color)

# Başlık ve etiketler
ax.set_title("Ackley Function (1D)", fontsize=18, fontweight='bold', loc='left', pad=20)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("f(x)", fontsize=14)

# Kenarlıkların kaldırılması ve grid ayarı
sns.despine(trim=True)
ax.grid(True, linestyle="--", alpha=0.3)

# Kenar boşluklarını optimize et
plt.tight_layout()
plt.show()
