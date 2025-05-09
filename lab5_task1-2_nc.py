


"""
Лабораторна робота №5: Візуалізація даних
Завдання 1-2: Інтерактивна візуалізація гармоніки з шумом та фільтрацією
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')  
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import signal


current_noise = None


def harmonic(t, amplitude, frequency, phase):
    """
    Функція генерує чисту гармоніку за формулою y(t) = A * sin(ω * t + φ)
    
    Параметри:
    t - часовий вектор
    amplitude - амплітуда (A)
    frequency - частота (ω)
    phase - фазовий зсув (φ)
    
    Повертає:
    Масив значень гармонічного сигналу
    """
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def generate_noise(t, noise_mean, noise_covariance):
    """
    Функція генерує нормально розподілений шум
    
    Параметри:
    t - часовий вектор (для визначення розміру шуму)
    noise_mean - середнє значення шуму
    noise_covariance - дисперсія шуму
    
    Повертає:
    Масив значень шуму
    """
    return np.random.normal(noise_mean, np.sqrt(noise_covariance), len(t))


def harmonic_with_noise(t, amplitude, frequency, phase=0, noise_mean=0, noise_covariance=0.1, show_noise=True, regenerate_noise=False):
    """
    Функція генерує гармоніку з накладеним шумом
    
    Параметри:
    t - часовий вектор
    amplitude - амплітуда гармоніки
    frequency - частота гармоніки
    phase - фазовий зсув гармоніки
    noise_mean - середнє значення шуму
    noise_covariance - дисперсія шуму
    show_noise - флаг відображення шуму
    regenerate_noise - флаг перегенерації шуму
    
    Повертає:
    Масив значень гармонічного сигналу з шумом або без
    """
    global current_noise
    
    
    pure_harmonic = harmonic(t, amplitude, frequency, phase)
    
    
    if not show_noise:
        return pure_harmonic
    
    
    if current_noise is None or regenerate_noise:
        current_noise = generate_noise(t, noise_mean, noise_covariance)
    
    
    return pure_harmonic + current_noise


def create_lowpass_filter(cutoff_freq, fs, order=5):
    """
    Створює фільтр нижніх частот Баттерворта
    
    Параметри:
    cutoff_freq - частота зрізу (Гц)
    fs - частота дискретизації (Гц)
    order - порядок фільтра
    
    Повертає:
    Коефіцієнти фільтра (b, a)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_filter(signal_data, cutoff_freq, fs, filter_type='butterworth', order=5, window_size=20):
    """
    Застосовує фільтр до сигналу
    
    Параметри:
    signal_data - вхідний сигнал для фільтрації
    cutoff_freq - частота зрізу для фільтрів частоти
    fs - частота дискретизації
    filter_type - тип фільтра ('butterworth', 'moving_average', 'median')
    order - порядок фільтра (для Баттерворта)
    window_size - розмір вікна (для ковзних середніх та медіанного фільтра)
    
    Повертає:
    Відфільтрований сигнал
    """
    if filter_type == 'butterworth':
        b, a = create_lowpass_filter(cutoff_freq, fs, order)
        filtered_data = signal.filtfilt(b, a, signal_data)
    
    elif filter_type == 'moving_average':
        window = np.ones(int(window_size)) / float(window_size)
        filtered_data = np.convolve(signal_data, window, 'same')
    
    elif filter_type == 'median':
        filtered_data = signal.medfilt(signal_data, kernel_size=int(window_size))
    
    else:
        raise ValueError(f"Невідомий тип фільтра: {filter_type}")
    
    return filtered_data

class HarmonicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Візуалізація гармоніки з шумом та фільтрацією")
        self.root.geometry("1000x800")
        
        
        self.initial_amplitude = 1.0
        self.initial_frequency = 1.0
        self.initial_phase = 0.0
        self.initial_noise_mean = 0.0
        self.initial_noise_covariance = 0.1
        self.initial_cutoff_freq = 2.0
        self.initial_filter_order = 5
        self.initial_window_size = 10
        self.show_noise = True
        self.filter_type = 'butterworth'
        
        
        self.t = np.linspace(0, 5, 1000)
        self.fs = 1 / (self.t[1] - self.t[0])  
        
        
        self.create_figure()
        
        
        self.create_controls()
        
        
        self.update_plot(regenerate=True)
    
    def create_figure(self):
        
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlabel('Час (t)')
        self.ax.set_ylabel('Амплітуда')
        self.ax.set_title('Візуалізація гармоніки з шумом та фільтрацією')
        self.ax.set_xlim([0, 5])
        self.ax.set_ylim([-3, 3])
        
        
        self.pure_line, = self.ax.plot([], [], 'g-', linewidth=1.5, label='Чиста гармоніка')
        self.noisy_line, = self.ax.plot([], [], 'r-', linewidth=0.7, label='Зашумлена гармоніка')
        self.filtered_line, = self.ax.plot([], [], 'b-', linewidth=1.5, label='Відфільтрована гармоніка')
        
        
        self.ax.legend(loc='upper right')
        
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    def create_controls(self):
        
        controls_frame = ttk.Frame(self.root, padding="10")
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        
        harmonic_frame = ttk.LabelFrame(controls_frame, text="Параметри гармоніки")
        harmonic_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        
        ttk.Label(harmonic_frame, text="Амплітуда:").grid(row=0, column=0, sticky='w')
        self.amplitude_var = tk.DoubleVar(value=self.initial_amplitude)
        self.amplitude_scale = ttk.Scale(harmonic_frame, from_=0.1, to=3.0, 
                                       variable=self.amplitude_var, orient=tk.HORIZONTAL, 
                                       length=200, command=self.on_parameter_change)
        self.amplitude_scale.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(harmonic_frame, textvariable=self.amplitude_var, width=5).grid(row=0, column=2)
        
        
        ttk.Label(harmonic_frame, text="Частота (Гц):").grid(row=1, column=0, sticky='w')
        self.frequency_var = tk.DoubleVar(value=self.initial_frequency)
        self.frequency_scale = ttk.Scale(harmonic_frame, from_=0.1, to=5.0, 
                                       variable=self.frequency_var, orient=tk.HORIZONTAL, 
                                       length=200, command=self.on_parameter_change)
        self.frequency_scale.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(harmonic_frame, textvariable=self.frequency_var, width=5).grid(row=1, column=2)
        
        
        ttk.Label(harmonic_frame, text="Фаза (рад):").grid(row=2, column=0, sticky='w')
        self.phase_var = tk.DoubleVar(value=self.initial_phase)
        self.phase_scale = ttk.Scale(harmonic_frame, from_=0.0, to=2*np.pi, 
                                   variable=self.phase_var, orient=tk.HORIZONTAL, 
                                   length=200, command=self.on_parameter_change)
        self.phase_scale.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(harmonic_frame, textvariable=self.phase_var, width=5).grid(row=2, column=2)
        
        
        noise_frame = ttk.LabelFrame(controls_frame, text="Параметри шуму")
        noise_frame.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        
        
        ttk.Label(noise_frame, text="Середнє шуму:").grid(row=0, column=0, sticky='w')
        self.noise_mean_var = tk.DoubleVar(value=self.initial_noise_mean)
        self.noise_mean_scale = ttk.Scale(noise_frame, from_=-1.0, to=1.0, 
                                        variable=self.noise_mean_var, orient=tk.HORIZONTAL, 
                                        length=200, command=self.on_noise_parameter_change)
        self.noise_mean_scale.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(noise_frame, textvariable=self.noise_mean_var, width=5).grid(row=0, column=2)
        
        
        ttk.Label(noise_frame, text="Дисперсія шуму:").grid(row=1, column=0, sticky='w')
        self.noise_cov_var = tk.DoubleVar(value=self.initial_noise_covariance)
        self.noise_cov_scale = ttk.Scale(noise_frame, from_=0.01, to=1.0, 
                                       variable=self.noise_cov_var, orient=tk.HORIZONTAL, 
                                       length=200, command=self.on_noise_parameter_change)
        self.noise_cov_scale.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(noise_frame, textvariable=self.noise_cov_var, width=5).grid(row=1, column=2)
        
        
        self.show_noise_var = tk.BooleanVar(value=self.show_noise)
        self.show_noise_check = ttk.Checkbutton(noise_frame, text="Показати шум", 
                                              variable=self.show_noise_var, 
                                              command=self.on_parameter_change)
        self.show_noise_check.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky='w')
        
        
        filter_frame = ttk.LabelFrame(controls_frame, text="Параметри фільтра")
        filter_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        
        ttk.Label(filter_frame, text="Тип фільтра:").grid(row=0, column=0, sticky='w')
        self.filter_type_var = tk.StringVar(value=self.filter_type)
        filter_types = {'Баттерворт': 'butterworth', 'Ковзне середнє': 'moving_average', 'Медіанний': 'median'}
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_type_var, 
                                  values=list(filter_types.keys()), state="readonly")
        filter_combo.grid(row=0, column=1, padx=5, pady=5)
        filter_combo.bind("<<ComboboxSelected>>", self.on_filter_type_change)
        
        
        ttk.Label(filter_frame, text="Частота зрізу (Гц):").grid(row=1, column=0, sticky='w')
        self.cutoff_var = tk.DoubleVar(value=self.initial_cutoff_freq)
        self.cutoff_scale = ttk.Scale(filter_frame, from_=0.1, to=10.0, 
                                    variable=self.cutoff_var, orient=tk.HORIZONTAL, 
                                    length=200, command=self.on_parameter_change)
        self.cutoff_scale.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(filter_frame, textvariable=self.cutoff_var, width=5).grid(row=1, column=2)
        
        
        ttk.Label(filter_frame, text="Порядок фільтра:").grid(row=2, column=0, sticky='w')
        self.order_var = tk.IntVar(value=self.initial_filter_order)
        self.order_scale = ttk.Scale(filter_frame, from_=1, to=10, 
                                   variable=self.order_var, orient=tk.HORIZONTAL, 
                                   length=200, command=self.on_parameter_change)
        self.order_scale.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(filter_frame, textvariable=self.order_var, width=5).grid(row=2, column=2)
        
        
        ttk.Label(filter_frame, text="Розмір вікна:").grid(row=3, column=0, sticky='w')
        self.window_var = tk.IntVar(value=self.initial_window_size)
        self.window_scale = ttk.Scale(filter_frame, from_=3, to=51, 
                                    variable=self.window_var, orient=tk.HORIZONTAL, 
                                    length=200, command=self.on_parameter_change)
        self.window_scale.grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(filter_frame, textvariable=self.window_var, width=5).grid(row=3, column=2)
        
        
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        
        ttk.Button(buttons_frame, text="Перегенерувати шум", 
                 command=lambda: self.update_plot(regenerate=True)).grid(row=0, column=0, padx=5)
        
        
        ttk.Button(buttons_frame, text="Скинути все", 
                 command=self.reset_parameters).grid(row=0, column=1, padx=5)
    
    def on_parameter_change(self, *args):
        self.update_plot()
    
    def on_noise_parameter_change(self, *args):
        
        self.update_plot(regenerate=True)
    
    def on_filter_type_change(self, *args):
        
        filter_types = {'Баттерворт': 'butterworth', 'Ковзне середнє': 'moving_average', 'Медіанний': 'median'}
        selected = self.filter_type_var.get()
        self.filter_type = filter_types.get(selected, 'butterworth')
        self.update_plot()
    
    def reset_parameters(self):
        
        self.amplitude_var.set(self.initial_amplitude)
        self.frequency_var.set(self.initial_frequency)
        self.phase_var.set(self.initial_phase)
        self.noise_mean_var.set(self.initial_noise_mean)
        self.noise_cov_var.set(self.initial_noise_covariance)
        self.cutoff_var.set(self.initial_cutoff_freq)
        self.order_var.set(self.initial_filter_order)
        self.window_var.set(self.initial_window_size)
        self.show_noise_var.set(True)
        self.filter_type_var.set('Баттерворт')
        self.filter_type = 'butterworth'
        self.update_plot(regenerate=True)
    
    def update_plot(self, regenerate=False):
        
        amplitude = self.amplitude_var.get()
        frequency = self.frequency_var.get()
        phase = self.phase_var.get()
        noise_mean = self.noise_mean_var.get()
        noise_cov = self.noise_cov_var.get()
        cutoff_freq = self.cutoff_var.get()
        filter_order = self.order_var.get()
        window_size = self.window_var.get()
        show_noise = self.show_noise_var.get()
        
        
        pure_signal = harmonic(self.t, amplitude, frequency, phase)
        noisy_signal = harmonic_with_noise(self.t, amplitude, frequency, phase, 
                                         noise_mean, noise_cov, show_noise, regenerate)
        
        
        if show_noise:
            filtered_signal = apply_filter(noisy_signal, cutoff_freq, self.fs, self.filter_type, 
                                         filter_order, window_size)
        else:
            filtered_signal = pure_signal
        
        
        self.pure_line.set_data(self.t, pure_signal)
        self.noisy_line.set_data(self.t, noisy_signal)
        self.noisy_line.set_visible(show_noise)
        self.filtered_line.set_data(self.t, filtered_signal)
        self.filtered_line.set_visible(show_noise)
        
        
        max_val = max(np.max(np.abs(pure_signal)), np.max(np.abs(noisy_signal)) if show_noise else 0)
        self.ax.set_ylim([-max_val*1.1, max_val*1.1])
        
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = HarmonicApp(root)
    root.mainloop()