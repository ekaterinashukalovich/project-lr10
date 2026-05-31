document.addEventListener('DOMContentLoaded', () => {
  // Элементы формы
  const forms = {
    predict: { el: document.getElementById('predict-form'), res: document.getElementById('predict-result') },
    csv:     { el: document.getElementById('csv-form'),     res: document.getElementById('csv-result') },
    model:   { el: document.getElementById('model-form'),   res: document.getElementById('model-result') }
  };

  // Отображение имени файла при выборе
  const csvInput = document.getElementById('csv-file');
  csvInput.addEventListener('change', (e) => {
    const fileName = e.target.files[0] ? e.target.files[0].name : '';
    document.getElementById('file-name').textContent = fileName;
  });

  // Вспомогательные функции
  const showResult = (container, message, type = 'info') => {
    container.className = `result ${type}`;
    container.innerHTML = message;
    container.classList.remove('hidden');
  };

  const toggleLoading = (formEl, loading) => {
    formEl.classList.toggle('loading', loading);
    const btn = formEl.querySelector('button');
    btn.disabled = loading;
    btn.textContent = loading ? '⏳ Обработка...' : btn.getAttribute('data-original-text') || btn.textContent;
  };

  // Сохраняем текст кнопок для восстановления
  document.querySelectorAll('form button').forEach(btn => {
    btn.setAttribute('data-original-text', btn.textContent);
  });

  // === 1. Предсказание для клиента (Сбор данных из формы) ===
  forms.predict.el.addEventListener('submit', async (e) => {
    e.preventDefault();
    toggleLoading(forms.predict.el, true);
    showResult(forms.predict.res, '⏳ Расчет вероятности...', 'info');

    try {
      // Собираем данные из всех полей
      const data = {
        person_age: parseFloat(document.getElementById('person_age').value),
        person_gender: document.getElementById('person_gender').value,
        person_education: document.getElementById('person_education').value,
        person_income: parseFloat(document.getElementById('person_income').value),
        person_emp_exp: parseInt(document.getElementById('person_emp_exp').value),
        person_home_ownership: document.getElementById('person_home_ownership').value,
        loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
        loan_intent: document.getElementById('loan_intent').value,
        loan_int_rate: parseFloat(document.getElementById('loan_int_rate').value),
        loan_percent_income: parseFloat(document.getElementById('loan_percent_income').value),
        cb_person_cred_hist_length: parseFloat(document.getElementById('cb_person_cred_hist_length').value),
        credit_score: parseInt(document.getElementById('credit_score').value),
        previous_loan_defaults_on_file: document.getElementById('previous_loan_defaults_on_file').value
      };

      // API ожидает массив объектов
      const payload = [data];

      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.error || 'Ошибка сервера');

      // Красивый вывод результата
      const pred = json.predictions[0];
      const statusClass = pred.loan_status === 1 ? 'success' : 'error';
      const statusText = pred.loan_status === 1 ? '✅ Кредит одобрен!' : '❌ Отказ в кредите';

      showResult(forms.predict.res, `
        <div style="text-align: center;">
          <h3 style="margin-bottom: 0.5rem; font-size: 1.2rem;">${statusText}</h3>
          <p>Вероятность одобрения: <strong>${(pred.probability_approved * 100).toFixed(2)}%</strong></p>
          <p style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">ID: ${pred.id || 'N/A'}</p>
        </div>
      `, statusClass);

    } catch (err) {
      showResult(forms.predict.res, `❌ ${err.message}`, 'error');
    } finally {
      toggleLoading(forms.predict.el, false);
    }
  });

  // === 2. CSV Предсказание ===
  forms.csv.el.addEventListener('submit', async (e) => {
    e.preventDefault();
    toggleLoading(forms.csv.el, true);
    showResult(forms.csv.res, '⏳ Загрузка и анализ данных...', 'info');

    const fd = new FormData();
    fd.append('file', csvInput.files[0]);

    try {
      const res = await fetch('/predict-from-csv', { method: 'POST', body: fd });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.error || 'Ошибка сервера');

      showResult(forms.csv.res, `
        <div style="text-align: center;">
          <h3>📊 Анализ завершен</h3>
          <p>Обработано записей: ${json.predictions_count}</p>
          ${json.roc_auc ? `<p>ROC-AUC: <strong>${json.roc_auc}</strong></p>` : ''}
          <p>Одобрено: ${json.approved_count} | Отказано: ${json.rejected_count}</p>
          ${json.download_link ? `<a href="${json.download_link}" target="_blank" style="color: var(--primary); font-weight: bold;">📥 Скачать результат</a>` : ''}
        </div>
      `, 'success');
    } catch (err) {
      showResult(forms.csv.res, ` ${err.message}`, 'error');
    } finally {
      toggleLoading(forms.csv.el, false);
    }
  });

  // === 3. Загрузка модели ===
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
      showResult(forms.model.res, `✅ ${json.message || 'Модель успешно загружена'}`, 'success');
    } catch (err) {
      showResult(forms.model.res, `❌ ${err.message}`, 'error');
    } finally {
      toggleLoading(forms.model.el, false);
    }
  });
});