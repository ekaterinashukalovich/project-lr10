document.addEventListener('DOMContentLoaded', () => {
  const forms = {
    predict: { el: document.getElementById('predict-form'), res: document.getElementById('predict-result') },
    csv:     { el: document.getElementById('csv-form'),     res: document.getElementById('csv-result') },
    model:   { el: document.getElementById('model-form'),   res: document.getElementById('model-result') }
  };

  const showResult = (container, message, type = 'info') => {
    container.className = `result ${type}`;
    container.innerHTML = message;
  };

  const toggleLoading = (formEl, loading) => {
    formEl.classList.toggle('loading', loading);
    formEl.querySelector('button').disabled = loading;
  };

  // 1. Predict
  forms.predict.el.addEventListener('submit', async (e) => {
    e.preventDefault();
    toggleLoading(forms.predict.el, true);
    showResult(forms.predict.res, ' Обработка...', 'info');

    try {
      const data = JSON.parse(document.getElementById('client-data').value);
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.error || 'Ошибка сервера');
      showResult(forms.predict.res, `<pre>${JSON.stringify(json, null, 2)}</pre>`, 'success');
    } catch (err) {
      showResult(forms.predict.res, ` ${err.message}`, 'error');
    } finally { toggleLoading(forms.predict.el, false); }
  });

  // 2. CSV Predict
  forms.csv.el.addEventListener('submit', async (e) => {
    e.preventDefault();
    toggleLoading(forms.csv.el, true);
    showResult(forms.csv.res, ' Загрузка и расчёт...', 'info');

    const fd = new FormData();
    fd.append('file', document.getElementById('csv-file').files[0]);

    try {
      const res = await fetch('/predict-from-csv', { method: 'POST', body: fd });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.error || 'Ошибка сервера');
      showResult(forms.csv.res, ` ROC-AUC: ${json.roc_auc}<br><a href="${json.download_link || '#'}" target="_blank">📥 Скачать результат</a>`, 'success');
    } catch (err) {
      showResult(forms.csv.res, ` ${err.message}`, 'error');
    } finally { toggleLoading(forms.csv.el, false); }
  });

  // 3. Upload Model
  forms.model.el.addEventListener('submit', async (e) => {
    e.preventDefault();
    toggleLoading(forms.model.el, true);
    showResult(forms.model.res, ' Загрузка модели...', 'info');

    const fd = new FormData();
    fd.append('file', document.getElementById('model-file').files[0]);

    try {
      const res = await fetch('/upload-model', { method: 'POST', body: fd });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.error || 'Ошибка сервера');
      showResult(forms.model.res, ` ${json.message || 'Модель загружена'}`, 'success');
    } catch (err) {
      showResult(forms.model.res, ` ${err.message}`, 'error');
    } finally { toggleLoading(forms.model.el, false); }
  });
});