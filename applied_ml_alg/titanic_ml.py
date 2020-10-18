import joblib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_age_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    '''Group passenger data by age.

    Parameters:
        df: A df of the titanic passenger data.
    '''
    df = (
        df
        .assign(age_group=-10)
        .fillna({'age': -10})
        .astype({'age': 'int32'})
    )
    age_groups = range(10, df.age.max() + 10, 10)
    full = df.query("age == -10")
    df = df.query("age != -10")
    for max_age in age_groups:
        tf = (
            df
            .query(f'age <= {max_age}')
            .assign(age_group=max_age)
        )
        full = full.append(tf, sort=False)
        df = df.query(f'age > {max_age}')

    full = (
        full
        .reset_index(drop=True)
        .sort_values(by=['age_group'])
    )
    return full


def show_relationships(df: pd.DataFrame, pass_vars: list):
    '''Show if variable is correlated with survival.

    Parameters:
        df: The titanic passenger df.
        pass_vars: The passenger variables to plot survival relationships for.
    '''
    titles = {
        'age_group': 'Passenger Age Group Compared to Survival',
        'sibsp': 'Number of Siblings or Spouses Compared to Survival',
        'parch': 'Number of Parents and Children Compared to Survival',
    }
    for i, pass_var in enumerate(pass_vars):
        tf = df
        if pass_var == 'age_group':
            tf = get_age_group_counts(df.copy())

        sns.catplot(x=pass_var, y='survived', data=tf, kind='point', aspect=2)
        plt.suptitle(titles[pass_var])

    plt.show()


def fill_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    '''Fill in missing ages based on sex and passenger class.

    Parameters:
        df: The passenger df.

    Returns:
        The passenger df with the missing ages assigned.
    '''
    df = df.fillna({'age': 0})
    tf = (
        df
        .query("age > 0")
        .reset_index(drop=True)
        .filter(['sex', 'pclass', 'age'])
        .groupby(by=['sex', 'pclass'], as_index=False)
        .median()
        .rename({'age': 'median_age'}, axis=1)
    )
    df = df.merge(tf, on=['sex', 'pclass'], how='left')
    df.loc[df.age == 0, 'age'] = df.median_age
    df = df.drop(labels=['median_age'], axis=1)
    return df


def prep_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    '''Combine and remove features for training the machine learning algorithm.

    Parameters:
        df: The titanic passenger df.

    Returns:
        The ml prepped df.
    '''
    df = fill_missing_age(df.copy())
    df['deck'] = df.deck.cat.add_categories(new_categories=['U'])
    df = (
        df
        .fillna({'deck': 'U'})
        .replace({
            'sex': {'male': 0, 'female': 1},
        })
        .assign(
            family_count=df.sibsp + df.parch
        )
        .filter(['survived', 'pclass', 'sex', 'age', 'family_count', 'fare'])
    )
    return df


def split_train_val_test(df: pd.DataFrame) -> dict:
    '''Split df for machine learning modeling.

    Parameters:
        df: The titanic passenger df.

    Returns:
        A dict with the split datasets.
    '''
    features = df.drop('survived', axis=1)
    labels = df.survived
    # First split the data into a 60% training set.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, random_state=42)

    # Then split the 40% chunk in half so you end up with 60% training, 20%
    # validation and 20% testting.
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42)
    ml_ds = {
        'features': features,
        'labels': labels,
        'x_train': x_train,
        'x_test': x_test,
        'x_val': x_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
    }
    return ml_ds


def get_db_tables(db_conn: sqlite3.Connection) -> pd.DataFrame:
    '''Load the tables in the sqlite database.

    Parameters:
        db_conn: The sqlite3 connection to the database.

    Returns:
        A df with the tables in the database.
    '''
    query = '''
    select
        type,
        name
    from sqlite_master
    where 1=1
        and type = 'table'
    '''
    df = pd.read_sql(query, db_conn)
    return df


def create_datasets(db_conn: sqlite3.Connection) -> dict:
    '''This loads the data sets from the sqlite database.

    Parameters:
        db_conn: The connection to the sqlite database.

    Returns:
        A dictionary of dfs.
    '''
    df = sns.load_dataset('titanic')
    # pass_vars = ['age_group', 'sibsp', 'parch']
    # show_relationships(df.copy(), pass_vars)
    df = prep_for_ml(df.copy())
    ml_ds = split_train_val_test(df.copy())
    for table_name, df in ml_ds.items():
        df.to_sql(con=db_conn, name=table_name, index=False,
                  if_exists='replace')

    return ml_ds


def load_datasets(db_conn: sqlite3.Connection) -> dict:
    '''This loads the titanic datasets.

    Parameters:
        db_conn: The connection to the sqlite database.

    Returns:
        A dictionary of the loaded dfs.
    '''
    # The ml datasets.
    exp_tables = ['features', 'lables', 'x_train', 'x_test', 'x_val',
                  'y_train', 'y_test', 'y_val']
    df = get_db_tables(db_conn)
    db_tables = df.name.unique().tolist()
    db_tables.sort()
    exp_tables.sort()
    if not db_tables == exp_tables:
        ml_ds = create_datasets(db_conn)
    else:
        ml_ds = load_datasets(db_conn, exp_tables)

    ml_ds = {}
    for table in exp_tables:
        query = f"select * from {table}"
        ml_ds[table] = pd.read_sql(query, db_conn)

    return ml_ds


def print_results(results: dict):
    '''Print the results of ml fit.

    Paramters:
        results: The results of the ml fit.
    '''
    print(f'Best params: {results.best_params_}\n')
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        mean = round(mean, 3)
        std = round(std * 2, 3)
        print(f'{mean} (+/- {std}) for {params}')


def get_logistic_estimator(ml_ds: dict):
    '''Runs the logistic regression and fit.

    Parameters:
        ml_ds: The data dict.
    '''
    lr = LogisticRegression()
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=5)
    cv.fit(ml_ds['features'], ml_ds['labels'])
    print_results(cv)
    print(cv.best_estimator_)


def main():
    '''The titanic machine learning problem. '''
    pd.set_option('display.max_columns', None)

    run_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    db_fn = run_dir.joinpath('titanic.db')
    with sqlite3.connect(db_fn) as db_conn:
        ml_ds = load_datasets(db_conn)

    get_logistic_estimator(ml_ds)


if __name__ == "__main__":
    main()
