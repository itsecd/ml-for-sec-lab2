# Машинное обучение для задач информационной безопасности.

В качестве стеганографической системы выбран метод НЗБ встраивание (запись по псевдослучайным координатам, 1-я битовая плоскость)

В качестве метода стеганоанализа - Метод длин серий (в качестве признаков - количество серий длин от 5 до 12)

Входными данными, необходимыми для выполнения лабораторной работы, являются 𝐾 полутоновых изображений одного размера (папка BOWS2).


Работа состоит из двух этапов:

1. Создание датасета (Lab2_Create_Dataset.ipynb)

- Реализована процедура расчёта векторов признаков, используемых для стегоанализа.

- Выполнена имитация работы стегосистемы для первых 𝐾2⁄ изображений -  Для различных значений q в каждое изображение в качестве стеганографической информации выла встроена отдельная реализация равномерного белого шума (число бит определялось текущим значением 𝑞). Вторая половина изображений не менялась.


2. Обучение

- Произведено обучение нескольких классификаторов по выборке, содержащей первые 70 % изображений каждого из двух типов (со встраиванием и без. То есть общий объём обучающей выборки составляет 𝐾∙0,7.

- Обученные классификаторы применены на оставшихся 30 % изображений и оценить качество классификации по мере Accuracy.

- Обучение проводилось для разных q и для разных наборов признаков.

-  Результат выведен в виде графиков зависимости Accuracy от 𝑞.








## Машинное обучение для задач информационной безопасности. Лаба 2. Порядок выполнения

## Схема сдачи

1. Сделать форк данного репозитория
2. Выбрать себе вариант задания и вписать его в табличку https://docs.google.com/spreadsheets/d/1Z1tym9FfX-Dj8huP2Q3iWVm0lrPpQz6NCGiweYGVy1w/edit?usp=sharing
3. Выполнить задание согласно выбранному варианту
4. Сделать pull request в данный репозиторий
5. Получить результат в рамках code review с замечаниями по коду.
6. При необходимости повторять пп. 3-4, пока преподаватель не отправит approve.
7. Во время онлайн-занятия защитить работу, ответить на вопросы преподавателя

## Более подробные рекомендации по работе с кодом

1. Форк *необходимо* сделать сразу. Для преподавателя это сигнализирует о том, что студент приступил к работе.
2. В описании репозитория нужно указать свои ФИО.
3. Желательно почаще делать коммиты. В идеале - как только решена некоторая промежуточная задача.
4. Коммиты *должны* иметь вменяемые описания.
5. Рекомендуется, чтобы ваш репозиторий содержал файлы [.gitignore](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files) (для них имеется набор [шаблонов](https://github.com/github/gitignore)) и [requirements.txt](https://www.jetbrains.com/help/pycharm/managing-dependencies.html#create-requirements)
