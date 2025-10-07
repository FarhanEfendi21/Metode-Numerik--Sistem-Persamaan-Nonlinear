"""
Penyelesaian Sistem Persamaan Non-Linear
NIM: 81
NIMx: 81 mod 4 = 1

Sistem Persamaan:
f1(x,y) = x² + xy - 10 = 0
f2(x,y) = y + 3xy² - 57 = 0

Kombinasi Fungsi Iterasi berdasarkan NIMx:
- NIMx = 0: g1A dan g2A
- NIMx = 1: g1A dan g2B  ← KOMBINASI NIMx : 1
- NIMx = 2: g1B dan g2A
- NIMx = 3: g1B dan g2B

Fungsi Iterasi:
g1A: x = (10 - x²)/y  (dari f1)
g1B: x = √(10 - xy)   (dari f1)
g2A: y = 57 - 3xy²    (dari f2)
g2B: y = √((57 - y)/(3x))  (dari f2)

Metode yang Digunakan:
1. Iterasi Titik Tetap - Jacobi (g1A dan g2B)
2. Iterasi Titik Tetap - Gauss-Seidel (g1A dan g2B)
3. Newton-Raphson
4. Secant
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# ==================== DEFINISI FUNGSI ====================

def f1(x, y):
    """f1(x,y) = x² + xy - 10"""
    return x**2 + x*y - 10

def f2(x, y):
    """f2(x,y) = y + 3xy² - 57"""
    return y + 3*x*y**2 - 57

# Turunan parsial untuk Newton-Raphson
def df1_dx(x, y):
    """∂f1/∂x = 2x + y"""
    return 2*x + y

def df1_dy(x, y):
    """∂f1/∂y = x"""
    return x

def df2_dx(x, y):
    """∂f2/∂x = 3y²"""
    return 3*y**2

def df2_dy(x, y):
    """∂f2/∂y = 1 + 6xy"""
    return 1 + 6*x*y

# ==================== FUNGSI ITERASI ====================

def g1A(x, y):
    """g1A: x = (10 - x²)/y"""
    if abs(y) < 1e-10:
        raise ValueError("Pembagi y mendekati nol")
    return (10 - x**2) / y

def g1B(x, y):
    """g1B: x = √(10 - xy)"""
    value = 10 - x*y
    if value < 0:
        raise ValueError("Nilai dalam akar negatif")
    return np.sqrt(value)

def g2A(x, y):
    """g2A: y = 57 - 3xy²"""
    return 57 - 3*x*y**2

def g2B(x, y):
    """g2B: y = √((57 - y)/(3x))"""
    if abs(x) < 1e-10:
        raise ValueError("Pembagi x mendekati nol")
    value = (57 - y) / (3*x)
    if value < 0:
        raise ValueError("Nilai dalam akar negatif")
    return np.sqrt(value)

# ==================== METODE ITERASI JACOBI ====================

def metode_jacobi(x0, y0, epsilon, g1_func, g2_func, func_name, max_iter=1000):
    """
    Metode Iterasi Titik Tetap - Jacobi (update simultan)
    x_{r+1} = g1(x_r, y_r)
    y_{r+1} = g2(x_r, y_r)  <- menggunakan x dan y lama
    """
    print("\n" + "="*80)
    print(f"METODE 1: ITERASI TITIK TETAP - JACOBI ({func_name})")
    print("="*80)
    print(f"Tebakan awal: x0 = {x0}, y0 = {y0}")
    print(f"Toleransi: ε = {epsilon}")
    print("Formula: x_{r+1} = g1(x_r, y_r), y_{r+1} = g2(x_r, y_r) [simultan]")
    print()
    
    iterations = []
    x, y = x0, y0
    
    for i in range(max_iter):
        try:
            # Hitung x dan y baru secara simultan (Jacobi)
            x_new = g1_func(x, y)
            y_new = g2_func(x, y)  # Menggunakan x dan y lama
            
            if not np.isfinite(x_new) or not np.isfinite(y_new):
                raise ValueError("Nilai tidak valid (inf/nan)")
            
            error = max(abs(x_new - x), abs(y_new - y))
            
            iterations.append({
                'Iterasi': i + 1,
                'x': x_new,
                'y': y_new,
                'f1(x,y)': f1(x_new, y_new),
                'f2(x,y)': f2(x_new, y_new),
                'Error': error
            })
            
            if abs(x_new) > 1e6 or abs(y_new) > 1e6:
                print(f"✗ DIVERGEN pada iterasi ke-{i+1} (nilai terlalu besar)")
                print(f"   x = {x_new:.6f}, y = {y_new:.6f}\n")
                df = pd.DataFrame(iterations[:min(10, len(iterations))])
                print("Beberapa Iterasi Awal:")
                print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
                return False, iterations, None, None
            
            if error < epsilon:
                print(f"✓ KONVERGEN pada iterasi ke-{i+1}")
                print(f"Solusi: x = {x_new:.10f}, y = {y_new:.10f}")
                print(f"Verifikasi: f1 = {f1(x_new, y_new):.2e}, f2 = {f2(x_new, y_new):.2e}\n")
                
                df = pd.DataFrame(iterations[-5:])
                print("5 Iterasi Terakhir:")
                print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.10f', showindex=False))
                return True, iterations, x_new, y_new
            
            x, y = x_new, y_new
            
        except Exception as e:
            print(f"✗ DIVERGEN atau ERROR pada iterasi ke-{i+1}: {str(e)}")
            if iterations:
                df = pd.DataFrame(iterations[:min(10, len(iterations))])
                print("\nBeberapa Iterasi Awal:")
                print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
            return False, iterations, None, None
    
    print(f"✗ Maksimum iterasi ({max_iter}) tercapai\n")
    return False, iterations, None, None

# ==================== METODE ITERASI GAUSS-SEIDEL ====================

def metode_seidel(x0, y0, epsilon, g1_func, g2_func, func_name, max_iter=1000):
    """
    Metode Iterasi Titik Tetap - Gauss-Seidel (update berurutan)
    x_{r+1} = g1(x_r, y_r)
    y_{r+1} = g2(x_{r+1}, y_r)  <- menggunakan x yang sudah diperbarui
    """
    print("\n" + "="*80)
    print(f"METODE 2: ITERASI TITIK TETAP - GAUSS-SEIDEL ({func_name})")
    print("="*80)
    print(f"Tebakan awal: x0 = {x0}, y0 = {y0}")
    print(f"Toleransi: ε = {epsilon}")
    print("Formula: x_{r+1} = g1(x_r, y_r), y_{r+1} = g2(x_{r+1}, y_r) [berurutan]")
    print()
    
    iterations = []
    x, y = x0, y0
    
    for i in range(max_iter):
        try:
            # Hitung x baru
            x_new = g1_func(x, y)
            if not np.isfinite(x_new):
                raise ValueError("Nilai x tidak valid (inf/nan)")
            
            # Hitung y baru menggunakan x_new (Gauss-Seidel)
            y_new = g2_func(x_new, y)
            if not np.isfinite(y_new):
                raise ValueError("Nilai y tidak valid (inf/nan)")
            
            error = max(abs(x_new - x), abs(y_new - y))
            
            iterations.append({
                'Iterasi': i + 1,
                'x': x_new,
                'y': y_new,
                'f1(x,y)': f1(x_new, y_new),
                'f2(x,y)': f2(x_new, y_new),
                'Error': error
            })
            
            # Cek divergensi
            if abs(x_new) > 1e6 or abs(y_new) > 1e6:
                print(f"✗ DIVERGEN pada iterasi ke-{i+1} (nilai terlalu besar)")
                print(f"   x = {x_new:.6f}, y = {y_new:.6f}\n")
                df = pd.DataFrame(iterations[:min(10, len(iterations))])
                print("Beberapa Iterasi Awal:")
                print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
                return False, iterations, None, None
            
            if error < epsilon:
                print(f"✓ KONVERGEN pada iterasi ke-{i+1}")
                print(f"Solusi: x = {x_new:.10f}, y = {y_new:.10f}")
                print(f"Verifikasi: f1 = {f1(x_new, y_new):.2e}, f2 = {f2(x_new, y_new):.2e}\n")
                
                df = pd.DataFrame(iterations[-5:])
                print("5 Iterasi Terakhir:")
                print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.10f', showindex=False))
                return True, iterations, x_new, y_new
            
            x, y = x_new, y_new
            
        except Exception as e:
            print(f"✗ DIVERGEN atau ERROR pada iterasi ke-{i+1}: {str(e)}")
            if iterations:
                df = pd.DataFrame(iterations[:min(10, len(iterations))])
                print("\nBeberapa Iterasi Awal:")
                print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.6f', showindex=False))
            return False, iterations, None, None
    
    print(f"✗ Maksimum iterasi ({max_iter}) tercapai\n")
    return False, iterations, None, None

# ==================== METODE NEWTON-RAPHSON ====================

def metode_newton_raphson(x0, y0, epsilon, max_iter=1000):
    """Metode Newton-Raphson"""
    print("\n" + "="*80)
    print("METODE 3: NEWTON-RAPHSON")
    print("="*80)
    print(f"Tebakan awal: x0 = {x0}, y0 = {y0}")
    print(f"Toleransi: ε = {epsilon}")
    print("Formula: Menggunakan matriks Jacobian dan turunan parsial")
    print()
    
    iterations = []
    x, y = x0, y0
    
    for i in range(max_iter):
        # Hitung matriks Jacobian
        J11 = df1_dx(x, y)
        J12 = df1_dy(x, y)
        J21 = df2_dx(x, y)
        J22 = df2_dy(x, y)
        
        det = J11 * J22 - J12 * J21
        
        if abs(det) < 1e-10:
            print("✗ DIVERGEN: Jacobian singular\n")
            return False, iterations, None, None
        
        F1 = f1(x, y)
        F2 = f2(x, y)
        
        # Hitung delta menggunakan inverse Jacobian
        dx = (J22 * F1 - J12 * F2) / det
        dy = (J11 * F2 - J21 * F1) / det
        
        x_new = x - dx
        y_new = y - dy
        
        error = max(abs(dx), abs(dy))
        
        # Hitung f1 dan f2 dengan nilai BARU
        f1_new = f1(x_new, y_new)
        f2_new = f2(x_new, y_new)
        
        iterations.append({
            'Iterasi': i + 1,
            'x': x_new,
            'y': y_new,
            'f1(x,y)': f1_new,
            'f2(x,y)': f2_new,
            'Error': error
        })
        
        if error < epsilon:
            print(f"✓ KONVERGEN pada iterasi ke-{i+1}")
            print(f"Solusi: x = {x_new:.10f}, y = {y_new:.10f}")
            print(f"Verifikasi: f1 = {f1(x_new, y_new):.2e}, f2 = {f2(x_new, y_new):.2e}\n")
            
            df = pd.DataFrame(iterations[-5:])
            print("5 Iterasi Terakhir:")
            print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.10f', showindex=False))
            return True, iterations, x_new, y_new
        
        x, y = x_new, y_new
    
    print(f"✗ Maksimum iterasi ({max_iter}) tercapai\n")
    return False, iterations, None, None

# ==================== METODE SECANT (DIPERBAIKI) ====================

def metode_secant(x0, y0, epsilon, max_iter=1000):
    """Metode Secant (aproksimasi turunan menggunakan finite difference)"""
    print("\n" + "="*80)
    print("METODE 4: SECANT")
    print("="*80)
    print(f"Tebakan awal: x0 = {x0}, y0 = {y0}")
    print(f"Toleransi: ε = {epsilon}")
    print("Formula: Aproksimasi Jacobian menggunakan finite difference")
    print()
    
    iterations = []
    x, y = x0, y0
    h = 1e-6  # Perturbasi kecil untuk aproksimasi turunan
    
    for i in range(max_iter):
        # Evaluasi fungsi di titik saat ini
        F1 = f1(x, y)
        F2 = f2(x, y)
        
        # Aproksimasi matriks Jacobian dengan finite difference
        # ∂f1/∂x ≈ [f1(x+h, y) - f1(x, y)] / h
        J11 = (f1(x + h, y) - F1) / h
        # ∂f1/∂y ≈ [f1(x, y+h) - f1(x, y)] / h
        J12 = (f1(x, y + h) - F1) / h
        # ∂f2/∂x ≈ [f2(x+h, y) - f2(x, y)] / h
        J21 = (f2(x + h, y) - F2) / h
        # ∂f2/∂y ≈ [f2(x, y+h) - f2(x, y)] / h
        J22 = (f2(x, y + h) - F2) / h
        
        # Determinan Jacobian
        det = J11 * J22 - J12 * J21
        
        if abs(det) < 1e-10:
            print("✗ DIVERGEN: Jacobian singular\n")
            return False, iterations, None, None
        
        # Hitung delta menggunakan inverse Jacobian (sama seperti Newton-Raphson)
        dx = (J22 * F1 - J12 * F2) / det
        dy = (J11 * F2 - J21 * F1) / det
        
        x_new = x - dx
        y_new = y - dy
        
        error = max(abs(dx), abs(dy))
        
        iterations.append({
            'Iterasi': i + 1,
            'x': x_new,
            'y': y_new,
            'f1(x,y)': f1(x_new, y_new),
            'f2(x,y)': f2(x_new, y_new),
            'Error': error
        })
        
        if error < epsilon:
            print(f"✓ KONVERGEN pada iterasi ke-{i+1}")
            print(f"Solusi: x = {x_new:.10f}, y = {y_new:.10f}")
            print(f"Verifikasi: f1 = {f1(x_new, y_new):.2e}, f2 = {f2(x_new, y_new):.2e}\n")
            
            df = pd.DataFrame(iterations[-5:])
            print("5 Iterasi Terakhir:")
            print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.10f', showindex=False))
            return True, iterations, x_new, y_new
        
        x, y = x_new, y_new
    
    print(f"✗ Maksimum iterasi ({max_iter}) tercapai\n")
    return False, iterations, None, None

# ==================== PROGRAM UTAMA ====================

def main():
    print("="*80)
    print("PENYELESAIAN SISTEM PERSAMAAN NON-LINEAR")
    print("="*80)
    print("NIM: 81")
    print("NIMx: 81 mod 4 = 1")
    print("\nSistem Persamaan:")
    print("  f1(x,y) = x² + xy - 10 = 0")
    print("  f2(x,y) = y + 3xy² - 57 = 0")
    print("\nKombinasi Fungsi Iterasi:")
    print("  NIMx = 0: g1A dan g2A")
    print("  NIMx = 1: g1A dan g2B  ← KOMBINASI UNTUK NIMx : 1")
    print("  NIMx = 2: g1B dan g2A")
    print("  NIMx = 3: g1B dan g2B")
    print("\nFungsi Iterasi:")
    print("  g1A: x = (10 - x²)/y")
    print("  g1B: x = √(10 - xy)")
    print("  g2A: y = 57 - 3xy²")
    print("  g2B: y = √((57 - y)/(3x))")
    print("="*80)
    
    # Parameter
    x0 = 1.5
    y0 = 3.5
    epsilon = 0.000001
    
    # Karena NIMx = 1, gunakan kombinasi g1A dan g2B
    print("\n" + "█"*80)
    print("KOMBINASI FUNGSI ITERASI: g1A dan g2B (NIMx = 1)")
    print("█"*80)
    
    results = {}
    
    # 1. Metode Jacobi dengan g1A dan g2B
    success, iters, x_sol, y_sol = metode_jacobi(x0, y0, epsilon, g1A, g2B, "g1A dan g2B")
    results['Jacobi (g1A+g2B)'] = {
        'success': success, 
        'iterations': len(iters) if iters else 0,
        'x': x_sol, 
        'y': y_sol
    }
    
    # 2. Metode Gauss-Seidel dengan g1A dan g2B
    success, iters, x_sol, y_sol = metode_seidel(x0, y0, epsilon, g1A, g2B, "g1A dan g2B")
    results['Seidel (g1A+g2B)'] = {
        'success': success, 
        'iterations': len(iters) if iters else 0,
        'x': x_sol, 
        'y': y_sol
    }
    
    # 3. Metode Newton-Raphson
    success, iters, x_sol, y_sol = metode_newton_raphson(x0, y0, epsilon)
    results['Newton-Raphson'] = {
        'success': success, 
        'iterations': len(iters) if iters else 0,
        'x': x_sol, 
        'y': y_sol
    }
    
    # 4. Metode Secant
    success, iters, x_sol, y_sol = metode_secant(x0, y0, epsilon)
    results['Secant'] = {
        'success': success, 
        'iterations': len(iters) if iters else 0,
        'x': x_sol, 
        'y': y_sol
    }
    
    # Ringkasan hasil
    print("\n" + "="*80)
    print("RINGKASAN HASIL SEMUA METODE")
    print("="*80)
    
    summary_data = []
    for method, result in results.items():
        if result['success']:
            summary_data.append({
                'Metode': method,
                'Status': '✓ Konvergen',
                'Iterasi': result['iterations'],
                'x': f"{result['x']:.10f}",
                'y': f"{result['y']:.10f}",
                'Error f1': f"{f1(result['x'], result['y']):.2e}",
                'Error f2': f"{f2(result['x'], result['y']):.2e}"
            })
        else:
            summary_data.append({
                'Metode': method,
                'Status': '✗ Divergen',
                'Iterasi': result['iterations'] if result['iterations'] > 0 else '-',
                'x': '-',
                'y': '-',
                'Error f1': '-',
                'Error f2': '-'
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(tabulate(df_summary, headers='keys', tablefmt='grid', showindex=False))
    
    print("\n" + "="*80)
    print("PENJELASAN METODE")
    print("="*80)
    print("1. JACOBI (g1A+g2B):")
    print("   - Update simultan: x_new dan y_new dihitung dari x_old dan y_old")
    print("   - x_{r+1} = g1A(x_r, y_r) = (10 - x_r²)/y_r")
    print("   - y_{r+1} = g2B(x_r, y_r) = √((57 - y_r)/(3x_r))")
    print("\n2. GAUSS-SEIDEL (g1A+g2B):")
    print("   - Update berurutan: y_new menggunakan x_new yang sudah dihitung")
    print("   - x_{r+1} = g1A(x_r, y_r) = (10 - x_r²)/y_r")
    print("   - y_{r+1} = g2B(x_{r+1}, y_r) = √((57 - y_r)/(3x_{r+1}))")
    print("\n3. NEWTON-RAPHSON:")
    print("   - Menggunakan matriks Jacobian dan turunan parsial eksplisit")
    print("   - Konvergensi kuadratik (sangat cepat)")
    print("\n4. SECANT:")
    print("   - Aproksimasi Jacobian menggunakan finite difference")
    print("   - Alternatif untuk Newton-Raphson tanpa turunan eksplisit")
    print("   - Menggunakan perturbasi h = 1e-6 untuk aproksimasi")
    print("="*80)
    
    print("\n" + "="*80)
    print("KESIMPULAN")
    print("="*80)
    print("✓ Solusi eksak sistem: x = 2, y = 3")
    print("  Verifikasi: f1(2,3) = 4 + 6 - 10 = 0 ✓")
    print("              f2(2,3) = 3 + 3(2)(9) - 57 = 3 + 54 - 57 = 0 ✓")
    print("\n✓ 4 Metode telah diimplementasikan:")
    print("  1. Iterasi Titik Tetap - Jacobi (g1A+g2B)")
    print("  2. Iterasi Titik Tetap - Gauss-Seidel (g1A+g2B)")
    print("  3. Newton-Raphson")
    print("  4. Secant (DIPERBAIKI)")
    print("\n✓ Kombinasi fungsi iterasi sesuai NIMx = 1: g1A dan g2B")
    print("\n✓ Perbedaan Newton-Raphson vs Secant:")
    print("  - Newton-Raphson: Turunan parsial dihitung secara eksplisit")
    print("  - Secant: Turunan parsial diaproksimasi dengan finite difference")
    print("="*80)

# Jalankan program
if __name__ == "__main__":
    main()
