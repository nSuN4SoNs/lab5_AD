from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot
from bokeh.models import (Slider, CheckboxGroup, Button, Select, 
                         ColumnDataSource, CustomJS, Div, RadioButtonGroup)
from bokeh.io import curdoc
import numpy as np



time_array = np.linspace(0, 10, 1000)
sampling_rate = 1 / (time_array[1] - time_array[0])


INIT_AMPLITUDE = 1.0
INIT_FREQUENCY = 0.5
INIT_PHASE = 0.0
INIT_NOISE_MEAN = 0.0
INIT_NOISE_VARIANCE = 0.1
INIT_FILTER_SIZE = 5


current_noise = None




def generate_harmonic(t, amplitude, frequency, phase):
    """Генерує гармонічний сигнал"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def generate_noise(t, mean, variance, seed=None):
    """Генерує шум із заданими параметрами"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, np.sqrt(variance), len(t))


def apply_noise_to_harmonic(t, amplitude, frequency, phase, noise_array=None, 
                           noise_mean=0, noise_variance=0.1):
    """Накладає шум на гармонічний сигнал"""
    harmonic = generate_harmonic(t, amplitude, frequency, phase)
    
    if noise_array is None:
        global current_noise
        current_noise = generate_noise(t, noise_mean, noise_variance)
        return harmonic + current_noise
    else:
        return harmonic + noise_array




def custom_median_filter(data, window_size):
    """Медіанний фільтр"""
    filtered_data = np.zeros_like(data)
    half_window = int(window_size // 2)
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        filtered_data[i] = np.median(data[start_idx:end_idx])
    
    return filtered_data


def custom_exponential_filter(data, alpha=0.2):
    """Експоненційний фільтр"""
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    
    return filtered_data


def custom_gaussian_filter(data, window_size, sigma=1.0):
    """Гаусівський фільтр"""
    filtered_data = np.zeros_like(data)
    half_window = int(window_size // 2)
    
    
    x = np.arange(-half_window, half_window + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  
    
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        kernel_start = max(0, half_window - i)
        kernel_end = min(len(kernel), len(kernel) - (i + half_window + 1 - len(data)))
        
        weighted_sum = np.sum(data[start_idx:end_idx] * kernel[kernel_start:kernel_end])
        weight_sum = np.sum(kernel[kernel_start:kernel_end])
        
        filtered_data[i] = weighted_sum / weight_sum
    
    return filtered_data





source_main = ColumnDataSource(data=dict(
    time=time_array,
    harmonic=generate_harmonic(time_array, INIT_AMPLITUDE, INIT_FREQUENCY, INIT_PHASE),
    noisy=apply_noise_to_harmonic(time_array, INIT_AMPLITUDE, INIT_FREQUENCY, INIT_PHASE, 
                                  noise_mean=INIT_NOISE_MEAN, noise_variance=INIT_NOISE_VARIANCE)
))


source_filtered = ColumnDataSource(data=dict(
    time=time_array,
    original=source_main.data['noisy'],
    median=custom_median_filter(source_main.data['noisy'], INIT_FILTER_SIZE),
    exponential=custom_exponential_filter(source_main.data['noisy']),
    gaussian=custom_gaussian_filter(source_main.data['noisy'], INIT_FILTER_SIZE)
))


fft_values = np.fft.rfft(source_main.data['noisy'])
fft_freq = np.fft.rfftfreq(len(time_array), d=time_array[1]-time_array[0])
source_fft = ColumnDataSource(data=dict(
    freq=fft_freq,
    amplitude=np.abs(fft_values)
))





plot_main = figure(title="Гармонічний сигнал з шумом", 
                  x_axis_label='Час (с)', y_axis_label='Амплітуда',
                  width=800, height=300, tools="pan,wheel_zoom,box_zoom,reset,save")
plot_main.background_fill_color = "


plot_main.line('time', 'harmonic', source=source_main, line_width=2, 
              line_color='blue', legend_label="Гармоніка")
plot_main.line('time', 'noisy', source=source_main, line_width=1.5, 
              line_color='red', legend_label="Сигнал з шумом")
plot_main.legend.location = "top_right"
plot_main.legend.click_policy = "hide"


plot_filtered = figure(title="Відфільтрований сигнал", 
                      x_axis_label='Час (с)', y_axis_label='Амплітуда',
                      width=800, height=300, tools="pan,wheel_zoom,box_zoom,reset,save",
                      x_range=plot_main.x_range)  
plot_filtered.background_fill_color = "


plot_filtered.line('time', 'original', source=source_filtered, line_width=1, 
                  line_color='red', alpha=0.5, legend_label="Оригінальний з шумом")
plot_filtered.line('time', 'median', source=source_filtered, line_width=2, 
                 line_color='green', legend_label="Медіанний фільтр")
plot_filtered.line('time', 'exponential', source=source_filtered, line_width=2, 
                  line_color='purple', legend_label="Експоненційний фільтр")
plot_filtered.line('time', 'gaussian', source=source_filtered, line_width=2, 
                  line_color='orange', legend_label="Гаусівський фільтр")
plot_filtered.legend.location = "top_right"
plot_filtered.legend.click_policy = "hide"


plot_fft = figure(title="Спектр сигналу", 
                 x_axis_label='Частота (Гц)', y_axis_label='Амплітуда',
                 width=800, height=300, tools="pan,wheel_zoom,box_zoom,reset,save")
plot_fft.background_fill_color = "


plot_fft.line('freq', 'amplitude', source=source_fft, line_width=1.5, line_color='blue')





controls_title = Div(text="<h3>Параметри сигналу та фільтрації</h3>")


slider_amplitude = Slider(title="Амплітуда", value=INIT_AMPLITUDE, start=0.1, end=5.0, step=0.1, width=250)
slider_frequency = Slider(title="Частота", value=INIT_FREQUENCY, start=0.1, end=5.0, step=0.1, width=250)
slider_phase = Slider(title="Фаза", value=INIT_PHASE, start=0, end=2*np.pi, step=0.1, width=250)


slider_noise_mean = Slider(title="Середнє шуму", value=INIT_NOISE_MEAN, start=-1.0, end=1.0, step=0.1, width=250)
slider_noise_variance = Slider(title="Дисперсія шуму", value=INIT_NOISE_VARIANCE, start=0.01, end=1.0, step=0.01, width=250)


slider_filter_window = Slider(title="Розмір вікна фільтра", value=INIT_FILTER_SIZE, start=3, end=31, step=2, width=250)


slider_exp_alpha = Slider(title="Альфа (експонен. фільтр)", value=0.2, start=0.01, end=0.99, step=0.01, width=250)


filter_select = Select(title="Відображати фільтр:", value="Всі", 
                     options=["Всі", "Медіанний", "Експоненційний", "Гаусівський"], width=250)


checkbox_show_noise = CheckboxGroup(labels=["Показувати шум"], active=[0], width=250)


layout_radio = RadioButtonGroup(labels=["Всі графіки", "Тільки сигнал", "Тільки фільтри", "Тільки FFT"], active=0)


button_regenerate_noise = Button(label="Перегенерувати шум", button_type="primary", width=250)
button_reset = Button(label="Скинути параметри", button_type="danger", width=250)




def update_harmonic():
    """Оновлює параметри гармонічного сигналу"""
    amplitude = slider_amplitude.value
    frequency = slider_frequency.value
    phase = slider_phase.value
    
    
    harmonic_values = generate_harmonic(time_array, amplitude, frequency, phase)
    source_main.data['harmonic'] = harmonic_values
    
    
    if 0 in checkbox_show_noise.active:  
        source_main.data['noisy'] = apply_noise_to_harmonic(
            time_array, amplitude, frequency, phase, 
            noise_array=current_noise
        )
    else:
        source_main.data['noisy'] = harmonic_values
    
    
    update_filters()
    
    
    update_fft()


def update_noise():
    """Оновлює параметри шуму"""
    amplitude = slider_amplitude.value
    frequency = slider_frequency.value
    phase = slider_phase.value
    noise_mean = slider_noise_mean.value
    noise_variance = slider_noise_variance.value
    
    
    global current_noise
    current_noise = generate_noise(time_array, noise_mean, noise_variance)
    
    
    if 0 in checkbox_show_noise.active:
        source_main.data['noisy'] = apply_noise_to_harmonic(
            time_array, amplitude, frequency, phase, 
            noise_array=current_noise
        )
    else:
        source_main.data['noisy'] = source_main.data['harmonic']
    
    
    update_filters()
    
    
    update_fft()


def update_filters():
    """Оновлює всі фільтри"""
    noisy_signal = source_main.data['noisy']
    window_size = slider_filter_window.value
    exp_alpha = slider_exp_alpha.value
    
    
    source_filtered.data.update({
        'original': noisy_signal,
        'median': custom_median_filter(noisy_signal, window_size),
        'exponential': custom_exponential_filter(noisy_signal, alpha=exp_alpha),
        'gaussian': custom_gaussian_filter(noisy_signal, window_size)
    })
    
    
    update_filter_visibility()


def update_filter_visibility():
    """Оновлює видимість фільтрів залежно від вибору у випадаючому меню"""
    selected_filter = filter_select.value
    
    if selected_filter == "Всі":
        
        plot_filtered.renderers[1].visible = True  
        plot_filtered.renderers[2].visible = True  
        plot_filtered.renderers[3].visible = True  
    else:
        
        plot_filtered.renderers[1].visible = False
        plot_filtered.renderers[2].visible = False
        plot_filtered.renderers[3].visible = False
        
        
        if selected_filter == "Медіанний":
            plot_filtered.renderers[1].visible = True
        elif selected_filter == "Експоненційний":
            plot_filtered.renderers[2].visible = True
        elif selected_filter == "Гаусівський":
            plot_filtered.renderers[3].visible = True


def update_fft():
    """Оновлює FFT графік"""
    noisy_signal = source_main.data['noisy']
    fft_values = np.fft.rfft(noisy_signal)
    
    source_fft.data.update({
        'amplitude': np.abs(fft_values)
    })


def update_noise_visibility():
    """Оновлює видимість шуму"""
    if 0 in checkbox_show_noise.active:  
        
        source_main.data['noisy'] = apply_noise_to_harmonic(
            time_array, slider_amplitude.value, slider_frequency.value, slider_phase.value, 
            noise_array=current_noise
        )
    else:
        
        source_main.data['noisy'] = source_main.data['harmonic']
    
    
    update_filters()
    
    
    update_fft()


def regenerate_noise():
    """Перегенерує шум"""
    update_noise()


def reset_parameters():
    """Скидає всі параметри до початкових значень"""
    
    slider_amplitude.value = INIT_AMPLITUDE
    slider_frequency.value = INIT_FREQUENCY
    slider_phase.value = INIT_PHASE
    slider_noise_mean.value = INIT_NOISE_MEAN
    slider_noise_variance.value = INIT_NOISE_VARIANCE
    slider_filter_window.value = INIT_FILTER_SIZE
    slider_exp_alpha.value = 0.2
    
    
    filter_select.value = "Всі"
    
    
    checkbox_show_noise.active = [0]
    
    
    layout_radio.active = 0
    
    
    regenerate_noise()
    
    
    update_layout_visibility()


def update_layout_visibility():
    """Оновлює видимість графіків залежно від вибору в радіокнопках"""
    selected_layout = layout_radio.active
    
    if selected_layout == 0:  
        plot_main.visible = True
        plot_filtered.visible = True
        plot_fft.visible = True
    elif selected_layout == 1:  
        plot_main.visible = True
        plot_filtered.visible = False
        plot_fft.visible = False
    elif selected_layout == 2:  
        plot_main.visible = False
        plot_filtered.visible = True
        plot_fft.visible = False
    elif selected_layout == 3:  
        plot_main.visible = False
        plot_filtered.visible = False
        plot_fft.visible = True





slider_amplitude.on_change('value', lambda attr, old, new: update_harmonic())
slider_frequency.on_change('value', lambda attr, old, new: update_harmonic())
slider_phase.on_change('value', lambda attr, old, new: update_harmonic())


slider_noise_mean.on_change('value', lambda attr, old, new: update_noise())
slider_noise_variance.on_change('value', lambda attr, old, new: update_noise())


slider_filter_window.on_change('value', lambda attr, old, new: update_filters())
slider_exp_alpha.on_change('value', lambda attr, old, new: update_filters())


filter_select.on_change('value', lambda attr, old, new: update_filter_visibility())


checkbox_show_noise.on_change('active', lambda attr, old, new: update_noise_visibility())


layout_radio.on_change('active', lambda attr, old, new: update_layout_visibility())


button_regenerate_noise.on_click(regenerate_noise)
button_reset.on_click(reset_parameters)





harmonic_controls = column(
    Div(text="<h4>Параметри гармоніки</h4>"),
    slider_amplitude,
    slider_frequency,
    slider_phase
)


noise_controls = column(
    Div(text="<h4>Параметри шуму</h4>"),
    slider_noise_mean,
    slider_noise_variance,
    checkbox_show_noise,
    button_regenerate_noise
)


filter_controls = column(
    Div(text="<h4>Параметри фільтрації</h4>"),
    slider_filter_window,
    slider_exp_alpha,
    filter_select
)


general_controls = column(
    Div(text="<h4>Загальні налаштування</h4>"),
    layout_radio,
    button_reset
)


controls = column(
    controls_title,
    row(harmonic_controls, noise_controls),
    row(filter_controls, general_controls)
)


plots = column(plot_main, plot_filtered, plot_fft)


layout = column(controls, plots)


curdoc().add_root(layout)
curdoc().title = "Інтерактивна візуалізація гармонічного сигналу"


regenerate_noise()
update_filters()
update_fft()
