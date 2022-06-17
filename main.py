import datetime
import json
import os
from functools import reduce

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cdrfilter import Filter
from cdrfilter.function import (
    SplitYearAsAge,
    ParseMileage,
    ParsePrice,
    RemoveLoanPrice,
    StandardizeBadges,
    OneHotEncodeDealRatings,
    OneHotEncodeBadges,
    SplitBrand,
    RemoveStockType,
    ClearModelNames,
    DealRatingsFromBadges,
)

DEAL_RATINGS = [
    'fairDeal',
    'goodDeal',
    'greatDeal',
]

PRED_CARS = [
    {
        "stockType": "Used",
        "title": "2020 Honda Civic LX",
        "mileage": "42,874 mi.",
        "primaryPrice": "$23,951",
        "loanPrice": "",
        "badges": [
            "Good Deal",
            "Home Delivery",
            "Virtual Appointments"
        ]
    },
    {
        "stockType": "Used",
        "title": "2018 BMW 430 i xDrive",
        "mileage": "28,598 mi.",
        "primaryPrice": "$34,990",
        "loanPrice": "",
        "badges": [
            "Good Deal",
            "Hot Car",
            "Home Delivery",
            "Virtual Appointments"
        ]
    },
]


def read_json_files():
    import glob

    path = os.path.join(os.getcwd(), 'data', 'json')
    json_files = glob.glob(os.path.join(path, "*.json"))

    arr = []
    for f in json_files:
        with open(f) as ff:
            arr.extend(json.load(ff))

    return arr


def get_all_badges(arr):
    badges = set()
    for f in arr:
        for b in f['badges']:
            badges.add(b)

    return badges


def get_models_freq(arr):
    models = {}
    for f in arr:
        title = f['model']
        if title in models:
            models[title] = models[title] + 1
        else:
            models[title] = 1

    return dict(sorted(models.items(), key=lambda x: x[1], reverse=True))


def filter_by_min_freq(arr, models_freq, thr):
    return [x for x in arr if models_freq[x['model']] >= thr]


def filter_by_min_price(arr, thr):
    return [x for x in arr if x['price'] >= thr]


def filter_by_min_mileage(arr, thr):
    return [x for x in arr if x['mileage'] >= thr]


def filter_by_deal_rating(arr):
    def has_value(x):
        values = [v for k, v in x.items() if k.startswith('dealRating')]
        return reduce(lambda a, b: a + b, values) == 1

    return [x for x in arr if has_value(x)]


def prep_json_data(data, **kwargs):
    def pipe(a, b):
        return a.pipe(b)

    filters = [
        SplitYearAsAge(),
        SplitBrand(),
        ParseMileage(),
        ParsePrice(),
        RemoveLoanPrice(),
        RemoveStockType(),
        StandardizeBadges(),
        DealRatingsFromBadges(DEAL_RATINGS),
        ClearModelNames(),
    ]
    data = reduce(pipe, filters, Filter(data)).apply()

    badges = kwargs.get('badges') or get_all_badges(data)
    filters = [
        OneHotEncodeDealRatings(DEAL_RATINGS),
        OneHotEncodeBadges(badges),
    ]

    data = reduce(pipe, filters, Filter(data)).apply()

    models_freq = kwargs.get('models_freq') or get_models_freq(data)
    data = filter_by_min_freq(data, models_freq, 10)
    data = filter_by_min_price(data, 1)
    data = filter_by_min_mileage(data, 1)
    data = filter_by_deal_rating(data)

    cb = kwargs.get('cb')
    if cb is not None:
        cb(badges=badges, models_freq=models_freq)

    return data


def json_to_df(data, serialize=True, s_path='data/data.csv'):
    df = pd.json_normalize(data)
    df = pd.get_dummies(df)

    if serialize:
        df.to_csv(s_path)

    return df


def split_x_y(df):
    delim = df.columns.str.startswith('dealRating')
    x = df.loc[:, ~delim]
    y = df.loc[:, delim]
    return x, y


def standard_scale(train, test):
    sc = StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)

    return train, test


def create_classifier():
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=15, activation='relu'),
        tf.keras.layers.Dense(units=15, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax'),
    ])

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )

    return classifier


def cross_validate(x, y, model_factory_fn, n_splits=5, **fit_args):
    weights_dir = os.path.join(os.getcwd(), 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    best_model = None
    best_acc = 0

    for k_fold, (train, test) in enumerate(KFold(n_splits=n_splits).split(x, y)):
        tf.keras.backend.clear_session()

        model = model_factory_fn()
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(),
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     metrics=tf.keras.metrics.CategoricalAccuracy(),
        # )

        x_train, x_test = standard_scale(x[train], x[test])

        log_dir = os.path.join(os.getcwd(), 'logs', 'fit', datetime.datetime.now().strftime('%Y%m%d-%H'), str(k_fold))
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        ho = model.fit(x_train, y[train], validation_data=(x_test, y[test]), callbacks=[tb_cb], **fit_args)
        model.save_weights(os.path.join(weights_dir, f'wg_{k_fold}.h5'))

        last_acc = ho.history['val_categorical_accuracy'][-1]
        if last_acc > best_acc:
            best_acc = last_acc
            best_model = model

    return best_model


def get_pred_df(**kwargs):
    df = kwargs.get('df')
    badges = kwargs.get('badges')
    models_freq = kwargs.get('models_freq')

    json_data = prep_json_data(PRED_CARS, badges=badges, models_freq=models_freq)
    df = pd.DataFrame(columns=df.columns)
    df = df.append(json_data).fillna(0)

    cols = ['model', 'brand']

    def encode(x):
        x = x.copy()
        for c in cols:
            x[f'{c}_{x[c]}'] = 1
        return x

    df = df.apply(encode, axis=1)
    df = df.drop(columns=cols)

    return split_x_y(df)[0]


def main():
    badges = []
    models_freq = {}

    def set_data_params(**kwargs):
        nonlocal badges, models_freq
        badges = kwargs.get('badges')
        models_freq = kwargs.get('models_freq')

    json_data = read_json_files()
    json_data = prep_json_data(json_data, cb=set_data_params)
    df = json_to_df(json_data, serialize=False)
    print(df.info())
    print(df.head())

    x, y = split_x_y(df)
    model = create_classifier()
    model.build(input_shape=x.shape)
    # model.load_weights(os.path.join(os.getcwd(), 'weights', 'wg_4.h5'))
    model = cross_validate(x.values, y.values, create_classifier, n_splits=5, batch_size=128, epochs=250, verbose=1)

    pred_df = get_pred_df(df=df, badges=badges, models_freq=models_freq)
    pred = model.predict(pred_df)

    print(pred)


if __name__ == '__main__':
    main()
