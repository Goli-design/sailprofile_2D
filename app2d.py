import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.interpolate import griddata, UnivariateSpline
import io
import re

# --- USTAWIENIA STRONY STREAMLIT ---
st.set_page_config(
    page_title="Analizator Profili Żagli 49er / FX (Matplotlib mm)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Wyłączenie ostrzeżeń Matplotlib dotyczących wątków
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- FUNKCJE POMOCNICZE (CAŁOŚĆ W MILIMETRACH [mm]) ---

def parse_and_clean_sail_mm(df_full):
    """
    Wczytuje dane żagla, gdzie wszystkie wartości (wysokość, odległość, głębokość, cięciwa) są w [mm].
    """
    chord_col = next((col for col in df_full.columns if 'chord' in col.lower()), None)
    if not chord_col:
        raise ValueError("Plik CSV musi zawierać kolumnę z długością cięciwy (np. 'Chord length').")
        
    chord_lengths = df_full[chord_col].copy()
    df_data = df_full.drop(columns=[chord_col])
    df_data.columns = pd.to_numeric(df_data.columns)
    
    return df_data, chord_lengths

def get_smooth_surface_2d_mm(df_data, chord_lengths, grid_x, grid_y):
    """
    Tworzy stabilną powierzchnię 2D żagla za pomocą interpolacji liniowej griddata (odpornej na błędy numeryczne).
    """
    leech_points = pd.DataFrame({'height': chord_lengths.index, 'distance': chord_lengths.values, 'depth': 0})
    df_stacked = df_data.stack().reset_index()
    df_stacked.columns = ['height', 'distance', 'depth']
    
    all_points = pd.concat([df_stacked, leech_points], ignore_index=True)
    
    # ZABEZPIECZENIE: Usuwanie duplikatów współrzędnych przed interpolacją 2D
    all_points = all_points.drop_duplicates(subset=['distance', 'height'], keep='first')
    
    points = all_points[['distance', 'height']].values
    values = all_points['depth'].values

    Z_grid = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # Płynna interpolacja liku wolnego (eliminacja schodków)
    for i, y_val in enumerate(grid_y[:, 0]):
        max_x = np.interp(y_val, chord_lengths.index, chord_lengths.values)
        Z_grid[i, grid_x[i, :] > max_x] = np.nan
            
    Z_grid[:, 0] = 0
    return Z_grid

def analyze_profile_geometry_mm(df_data, chord_lengths):
    """
    Oblicza 8 parametrów aerodynamicznych profilu dla każdej wysokości żagla.
    Wszystkie obliczenia i dane wejściowe/wyjściowe są w [mm].
    """
    results = []
    for height_mm, profile in df_data.iterrows():
        profile_clean = profile.dropna()
        x_measured = profile_clean.index.values.astype(float)
        z_measured = profile_clean.values
        chord_mm = chord_lengths.loc[height_mm]
        
        # ZABEZPIECZENIE: Odrzucenie punktów leżących poza długością cięciwy
        valid_mask = x_measured < chord_mm
        x_measured = x_measured[valid_mask]
        z_measured = z_measured[valid_mask]
        
        x_complete = np.append(x_measured, chord_mm)
        z_complete = np.append(z_measured, 0)
        sort_idx = np.argsort(x_complete)
        
        x_sorted = x_complete[sort_idx]
        z_sorted = z_complete[sort_idx]
        
        # Dynamiczne dopasowanie stopnia krzywej spline do liczby punktów (zapobiega Singular Matrix)
        num_pts = len(x_sorted)
        k_degree = min(3, num_pts - 1)
        
        if k_degree >= 1:
            spline = UnivariateSpline(x_sorted, z_sorted, s=0, k=k_degree)
        else:
            continue
        
        x_fine = np.linspace(0, chord_mm, 2000)
        z_fine = spline(x_fine)
        
        max_depth_mm = np.max(z_fine)
        max_depth_pos_mm = x_fine[np.argmax(z_fine)]

        if chord_mm > 0:
            max_depth_perc_chord = (max_depth_mm / chord_mm) * 100
            max_depth_pos_perc_chord = (max_depth_pos_mm / chord_mm) * 100
        else:
            max_depth_perc_chord = max_depth_pos_perc_chord = 0
        
        x_front_mid = max_depth_pos_mm / 2
        front_depth_mm = spline(x_front_mid)

        x_rear_mid = max_depth_pos_mm + (chord_mm - max_depth_pos_mm) / 2
        rear_depth_mm = spline(x_rear_mid)

        if max_depth_mm > 0:
            front_depth_perc_max = (front_depth_mm / max_depth_mm) * 100
            rear_depth_perc_max = (rear_depth_mm / max_depth_mm) * 100
        else:
            front_depth_perc_max = rear_depth_perc_max = 0
        
        spline_deriv = spline.derivative(n=1)
        slope_entry = spline_deriv(0)
        entry_angle_deg = np.degrees(np.arctan(slope_entry))

        slope_exit = spline_deriv(chord_mm)
        exit_angle_deg = np.degrees(np.arctan(slope_exit))

        results.append({
            'Wysokość (mm)': height_mm,
            'Maks. głębokość (% cięciwy)': round(max_depth_perc_chord, 1),
            'Poz. maks. głębokości (% cięciwy)': round(max_depth_pos_perc_chord, 1),
            'Głęb. przednia (% maks.)': round(front_depth_perc_max, 1),
            'Głęb. tylna (% maks.)': round(rear_depth_perc_max, 1),
            'Kąt natarcia (stopnie)': round(entry_angle_deg, 1),
            'Kąt spływu (stopnie)': round(exit_angle_deg, 1)
        })
        
    return pd.DataFrame(results).set_index('Wysokość (mm)')

# --- INTERFEJS UŻYTKOWNIKA ---

st.title("⛵ Aerodynamiczny Analizator i Komparator Żagli [Skala mm]")
st.markdown("Narzędzie dedykowane dla klas **49er** oraz **49er FX**. Porównuje dwa projekty żagli (Wizualizacja 3D Matplotlib) oraz oblicza parametry profili.")

# Panel boczny - Przesyłanie plików
st.sidebar.header("📁 Wczytywanie danych")
orig_file = st.sidebar.file_uploader("Wybierz żagiel ORYGINALNY (CSV w mm)", type="csv")
mod_file = st.sidebar.file_uploader("Wybierz żagiel ZMODYFIKOWANY (CSV w mm)", type="csv")

if orig_file and mod_file:
    # Wczytanie plików wejściowych
    df_orig_raw = pd.read_csv(orig_file, sep=';', decimal=',', index_col=0)
    df_mod_raw = pd.read_csv(mod_file, sep=';', decimal=',', index_col=0)
    
    orig_name = orig_file.name.replace('.csv', '')
    mod_name = mod_file.name.replace('.csv', '')
    
    try:
        # Przetwarzanie danych
        df_orig, chords_orig = parse_and_clean_sail_mm(df_orig_raw)
        df_mod, chords_mod = parse_and_clean_sail_mm(df_mod_raw)
        
        # Obliczenie maksymalnych wymiarów siatki głównej (w całości w [mm])
        max_chord = max(chords_orig.max(), chords_mod.max())
        max_height = max(df_orig.index.max(), df_mod.index.max())
        min_height = min(df_orig.index.min(), df_mod.index.min())

        # Zagęszczona siatka obliczeniowa (krok 10 mm poziom, 20 mm pion)
        x_master = np.arange(0, max_chord + 10, 10)
        y_master = np.arange(min_height, max_height + 20, 20)
        X_master, Y_master = np.meshgrid(x_master, y_master)

        # Wygładzanie 2D powierzchni żagli
        Z_orig = get_smooth_surface_2d_mm(df_orig, chords_orig, X_master, Y_master)
        Z_mod = get_smooth_surface_2d_mm(df_mod, chords_mod, X_master, Y_master)
        
        global_max_depth = np.nanmax([Z_orig, Z_mod])
        Z_diff = Z_mod - Z_orig
        max_abs_diff = np.nanmax(np.abs(Z_diff))

        # Obliczenia parametrów 2D
        table_orig = analyze_profile_geometry_mm(df_orig, chords_orig)
        table_mod = analyze_profile_geometry_mm(df_mod, chords_mod)

        # --- ZAKŁADKI W INTERFEJSIE ---
        tab1, tab2, tab3 = st.tabs(["📊 Porównanie 3D [mm]", "🔍 Wykres Różnicowy 3D [mm]", "📋 Parametry & Raport Excel"])

        with tab1:
            st.header("Porównanie geometrii żagli (Wygładzone modele 3D)")
            st.write("Wizualizacja wygenerowana przy zachowaniu proporcji rzeczywistych z dwukrotnym (2x) powiększeniem głębokości (Z). Wszystkie osie w [mm].")
            
            # Tworzenie stabilnego wykresu Matplotlib dla obu żagli
            fig_comp = plt.figure(figsize=(18, 8))
            
            # --- ŻAGIEL 1: ORYGINAŁ ---
            ax1 = fig_comp.add_subplot(1, 2, 1, projection='3d')
            surf1 = ax1.plot_surface(X_master, Y_master, Z_orig, cmap='viridis', 
                                     vmin=0, vmax=global_max_depth, edgecolor='none', alpha=0.9)
            ax1.set_title(f'Oryginalny: {orig_name}', fontsize=14, pad=20)
            # Box aspect z dokładnie dwukrotnym powiększeniem głębokości w skali mm
            ax1.set_box_aspect((np.nanmax(X_master), np.nanmax(Y_master), global_max_depth * 2.0))
            ax1.view_init(elev=25., azim=-135)
            ax1.set_xlabel('Odległość (mm)')
            ax1.set_ylabel('Wysokość (mm)')
            ax1.set_zlabel("Głębokość (mm)")

            # --- ŻAGIEL 2: MODYFIKACJA ---
            ax2 = fig_comp.add_subplot(1, 2, 2, projection='3d')
            surf2 = ax2.plot_surface(X_master, Y_master, Z_mod, cmap='viridis', 
                                     vmin=0, vmax=global_max_depth, edgecolor='none', alpha=0.9)
            ax2.set_title(f'Zmodyfikowany: {mod_name}', fontsize=14, pad=20)
            ax2.set_box_aspect((np.nanmax(X_master), np.nanmax(Y_master), global_max_depth * 2.0))
            ax2.view_init(elev=25., azim=-135)
            ax2.set_xlabel('Odległość (mm)')
            ax2.set_ylabel('Wysokość (mm)')
            ax2.set_zlabel("Głębokość (mm)")

            # Estetyczne ułożenie legendy bez nachodzenia na wykresy
            fig_comp.subplots_adjust(right=0.85)
            cbar_ax = fig_comp.add_axes([0.88, 0.25, 0.02, 0.5])
            cbar = fig_comp.colorbar(surf2, cax=cbar_ax)
            cbar.set_label('Głębokość profilu (mm)', size=12)

            st.pyplot(fig_comp)

        with tab2:
            st.header("Trójwymiarowy Wykres Różnicowy")
            st.write("Czerwony kolor oznacza miejsca, gdzie żagiel zmodyfikowany jest głębszy. Niebieski - gdzie jest płaski. Wszystkie osie w [mm].")
            
            fig_diff = plt.figure(figsize=(11, 9))
            ax3 = fig_diff.add_subplot(1, 1, 1, projection='3d')
            ax3.set_title(f'Różnica 3D: "{mod_name}" vs "{orig_name}"', fontsize=14, pad=20)

            # Normalizacja kolorów od -max_diff do +max_diff
            norm = mcolors.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(norm(Z_diff))

            # Żagiel oryginalny jako lekka, szara referencja pod spodem
            ax3.plot_surface(X_master, Y_master, Z_orig, color='grey', alpha=0.15, edgecolor='none')
            
            # Żagiel zmodyfikowany pomalowany kolorami różnic
            diff_surf = ax3.plot_surface(X_master, Y_master, Z_mod, facecolors=colors, 
                                         linewidth=0, antialiased=True, shade=False, alpha=0.85)

            ax3.set_box_aspect((np.nanmax(X_master), np.nanmax(Y_master), global_max_depth * 2.0))
            ax3.view_init(elev=25., azim=-135)
            ax3.set_xlabel('Odległość (mm)')
            ax3.set_ylabel('Wysokość (mm)')
            ax3.set_zlabel("Głębokość (mm)")

            # Dodanie bocznej legendy różnic
            m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(Z_diff)
            cbar_diff = fig_diff.colorbar(m, shrink=0.6, aspect=20, pad=0.05, ax=ax3)
            cbar_diff.set_label(f'Różnica głębokości (mm, {mod_name} - {orig_name})', size=11)

            st.pyplot(fig_diff)

        with tab3:
            st.header("Analiza Parametryczna Profili")
            
            # Generowanie skoroszytu Excel w pamięci RAM serwera
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Oczyszczanie i skracanie nazw arkuszy do limitu Excela (30 znaków)
                sheet_orig = re.sub(r'[\\/*?:\[\]]', '', orig_name)[:30]
                sheet_mod = re.sub(r'[\\/*?:\[\]]', '', mod_name)[:30]
                
                table_orig.to_excel(writer, sheet_name=sheet_orig)
                table_mod.to_excel(writer, sheet_name=sheet_mod)
                
            st.download_button(
                label="📥 Pobierz wyniki w jednym pliku Excel (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"analiza_porownawcza_{orig_name}_vs_{mod_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.subheader(f"Oryginał: {orig_name}")
                st.dataframe(table_orig)
                
            with col_t2:
                st.subheader(f"Modyfikacja: {mod_name}")
                st.dataframe(table_mod)

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania plików. Upewnij się, że oba pliki posiadają prawidłową strukturę. Szczegóły błędu: {e}")

else:
    # Komunikat startowy, gdy pliki nie zostały jeszcze wczytane
    st.info("👈 Aby rozpocząć analizę, prześlij oba pliki CSV (Oryginalny oraz Zmodyfikowany w skali mm) w panelu bocznym po lewej stronie.")