import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import typer
from dataclasses import dataclass
from darts import TimeSeries
from darts.models import Prophet
import holidays
import optuna
import logging
from plotly.subplots import make_subplots
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


app = typer.Typer()



@dataclass
class Config:
    steps_days: int = 60
    last_n_days_profile: int = 60
    use_crossvalidation: bool = False
    use_params_tuning: bool = False


def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX are supported")



def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    return mape, wape


def objective(trial, series, holidays, config):

    params = {
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10),
    }

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        **params
    )

    horizon = config.steps_days

    train = series[:-horizon]
    val = series[-horizon:]

    holidays_train = holidays[:len(train)]
    holidays_val = holidays[len(train):]

    model.fit(train, future_covariates=holidays_train)
    pred = model.predict(n=horizon, future_covariates=holidays_val)

    y_true = val.values().flatten()
    y_pred = pred.values().flatten()

    _, wape = calculate_metrics(y_true, y_pred)

    return wape


def plot_wape_history(study, ts_name, output_dir):
    """Построение графика изменения WAPE в процессе оптимизации"""

    if study is None:
        return

    # Получаем историю trials
    trials_df = study.trials_dataframe()

    if len(trials_df) == 0:
        return

    # Фильтруем только завершенные trials
    complete_trials = trials_df[trials_df['state'] == 'COMPLETE']

    if len(complete_trials) == 0:
        return

    # Создаем график
    fig = go.Figure()

    # График значений WAPE по trials (точки)
    fig.add_trace(
        go.Scatter(
            x=complete_trials['number'],
            y=complete_trials['value'],
            mode='markers',
            name='WAPE value',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.6
            ),
            text=[f"Trial {i}<br>WAPE: {wape:.2f}%" for i, wape in
                  zip(complete_trials['number'], complete_trials['value'])],
            hoverinfo='text'
        )
    )

    # Линия, соединяющая точки
    fig.add_trace(
        go.Scatter(
            x=complete_trials['number'],
            y=complete_trials['value'],
            mode='lines',
            name='WAPE trend',
            line=dict(color='blue', width=1, dash='solid'),
            showlegend=True
        )
    )

    # Лучшее значение на данный момент (кумулятивный минимум)
    best_values = complete_trials['value'].cummin()
    fig.add_trace(
        go.Scatter(
            x=complete_trials['number'],
            y=best_values,
            mode='lines',
            name='Best WAPE so far',
            line=dict(color='red', width=2, dash='dash')
        )
    )

    # Отмечаем лучший trial
    best_trial_idx = complete_trials['value'].idxmin()
    best_trial = complete_trials.loc[best_trial_idx]
    best_wape = best_trial['value']
    best_number = best_trial['number']

    fig.add_trace(
        go.Scatter(
            x=[best_number],
            y=[best_wape],
            mode='markers',
            name=f'Best trial (WAPE: {best_wape:.2f}%)',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(color='black', width=1)
            )
        )
    )

    # Настройка оформления
    fig.update_layout(
        title=dict(
            text=f"WAPE Optimization History - {ts_name}<br><sub>Best WAPE: {best_wape:.2f}% | Total trials: {len(complete_trials)}</sub>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title="Trial Number",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="WAPE (%)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        height=500,
        width=900
    )

    # Сохраняем график
    plot_path = os.path.join(output_dir, f"{ts_name}_wape_history.html")
    fig.write_html(plot_path)
    logger.info(f"Saved WAPE history plot: {plot_path}")



def preprocess(df, config: Config):
    df = df.copy()
    df['dttm_30'] = pd.to_datetime(df['dttm_30'])

    def get_intraday_profile(df_history):
        last_date = df_history['dttm_30'].max()
        start_date = last_date - pd.Timedelta(days=config.last_n_days_profile)

        recent_data = df_history[df_history['dttm_30'] > start_date].copy()
        recent_data['time'] = recent_data['dttm_30'].dt.time
        recent_data['is_weekend'] = recent_data['dttm_30'].dt.weekday >= 5

        profile = (
            recent_data
            .groupby(['is_weekend', 'time'])['y']
            .mean()
            .reset_index()
        )

        # нормализация внутри каждой группы
        profile['share'] = profile.groupby('is_weekend')['y'].transform(
            lambda x: x / x.sum() if x.sum() != 0 else 1 / 48
        )

        return profile[['is_weekend', 'time', 'share']]

    def aggregate_to_daily(df):
        daily = df.copy()
        daily['date'] = daily['dttm_30'].dt.date
        daily_agg = daily.groupby('date')['y'].sum().reset_index()
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        return daily_agg

    def get_holiday_regressor(df_dates):
        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        df = pd.DataFrame({'date': pd.to_datetime(df_dates.unique())})
        df['is_holiday'] = df['date'].apply(
            lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0
        )
        return df

    # добавляем будущие даты
    max_date = df['dttm_30'].max().date()
    future_dates = pd.date_range(
        start=max_date,
        periods=60,  # запас
        freq='D'
    )

    all_dates = pd.concat([
        pd.Series(df['dttm_30'].dt.date),
        pd.Series(future_dates.date)
    ])

    holiday_df = get_holiday_regressor(all_dates)
    holiday_series = TimeSeries.from_dataframe(holiday_df, 'date', 'is_holiday', freq='D')

    return df, get_intraday_profile, aggregate_to_daily, holiday_series



def forecast_daily(series_daily, holidays_train, future_holidays, config: Config, ts_name, output_dir):
    best_params = {}
    study = None
    if config.use_params_tuning:
        study = optuna.create_study(direction="minimize")
        def log_trial(study, trial):
            logger.info(f"Trial {trial.number}: WAPE = {trial.value:.4f}% | "
                        f"Params: {trial.params}")
        study.optimize(
            lambda trial: objective(trial, series_daily, holidays_train, config),
            n_trials=20,
            callbacks=[log_trial],
        )
        best_params = study.best_params
        logger.info(f"Best params for {ts_name}: {best_params} | Best WAPE: {study.best_value:.4f}%")
        wape_plot_dir = os.path.join(output_dir, "wape_analysis")
        os.makedirs(wape_plot_dir, exist_ok=True)
        plot_wape_history(study, ts_name, wape_plot_dir)
    else:
        best_params = {}
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        **best_params
    )
    if config.use_crossvalidation:
        score = cross_validate(series_daily, holidays_train, config)
        logger.info(f"CV WAPE: {score}")

    model.fit(series_daily, future_covariates=holidays_train)
    metrics = {
        "cv_wape": score if config.use_crossvalidation else None,
        "best_wape": study.best_value if study else None
    }
    return model.predict(
        n=config.steps_days,
        future_covariates=future_holidays
    ), best_params, metrics, study


def disaggregate_to_intraday(fcst_daily, profile, start_time, steps_days):
    df_fcst_daily = pd.DataFrame({
        'date': pd.to_datetime(fcst_daily.time_index.date),
        'y_daily': fcst_daily.values().flatten()
    })
    future_index = pd.date_range(
        start=start_time,
        periods=steps_days * 48,
        freq='30min'
    )
    fcst_30 = pd.DataFrame({'dttm_30': future_index})
    fcst_30['date'] = fcst_30['dttm_30'].dt.floor('D')
    fcst_30['time'] = fcst_30['dttm_30'].dt.time
    fcst_30['is_weekend'] = fcst_30['dttm_30'].dt.weekday >= 5
    fcst_30 = fcst_30.merge(df_fcst_daily, on='date', how='left')
    fcst_30 = fcst_30.merge(profile, on=['is_weekend', 'time'], how='left')
    fcst_30['forecast'] = fcst_30['y_daily'] * fcst_30['share']
    fcst_30['forecast'] = fcst_30['forecast'].clip(lower=0)
    return fcst_30[['dttm_30', 'forecast']]


def get_new_year_profile(df):

    df = df.copy()

    df['date'] = pd.to_datetime(df['dttm_30']).dt.date

    daily = (
        df.groupby('date')['y']
        .sum()
        .reset_index()
    )

    daily['date'] = pd.to_datetime(daily['date'])

    profiles = []

    for year in [2023, 2024, 2025]:

        # праздничные дни
        ny_range = pd.date_range(
            f"{year}-01-01",
            f"{year}-01-08"
        )

        # обычные будни ДО нового года
        baseline_df = daily[
            (daily['date'] >= f"{year-1}-12-15") &
            (daily['date'] < f"{year}-01-01")
        ].copy()

        # только будние
        baseline_df = baseline_df[
            baseline_df['date'].dt.weekday < 5
        ]

        baseline = baseline_df['y'].mean()

        if pd.isna(baseline) or baseline == 0:
            continue

        # коэффициенты для каждого дня
        for d in ny_range:

            val = daily[daily['date'] == d]['y']

            if len(val) == 0:
                continue

            coef = val.values[0] / baseline

            profiles.append({
                "month_day": d.strftime("%m-%d"),
                "coef": coef
            })

    profile_df = pd.DataFrame(profiles)

    # усредняем коэффициенты по годам
    final_profile = (
        profile_df
        .groupby('month_day')['coef']
        .mean()
        .to_dict()
    )

    return final_profile



def apply_new_year_adjustment(fcst_df, ny_profile):
    fcst_df = fcst_df.copy()

    fcst_df['date'] = pd.to_datetime(fcst_df['dttm_30']).dt.date

    # daily forecast
    daily_fcst = (
        fcst_df
        .groupby('date')['forecast']
        .sum()
        .reset_index()
    )

    daily_fcst['date'] = pd.to_datetime(daily_fcst['date'])

    years = daily_fcst['date'].dt.year.unique()

    for year in years:

        # проверяем есть ли НГ в прогнозе
        ny_days = pd.date_range(
            f"{year}-01-01",
            f"{year}-01-08"
        )

        if not daily_fcst['date'].isin(ny_days).any():
            continue

        # 2 недели до НГ
        baseline_df = daily_fcst[
            (daily_fcst['date'] >= f"{year-1}-12-15") &
            (daily_fcst['date'] < f"{year}-01-01")
        ].copy()

        # только будни
        baseline_df = baseline_df[
            baseline_df['date'].dt.weekday < 5
        ]

        if len(baseline_df) == 0:
            continue

        X = baseline_df['forecast'].mean()


        for d in ny_days:

            md = d.strftime("%m-%d")

            if md not in ny_profile:
                continue

            target_daily_value = X * ny_profile[md]

            # заменяем дневной прогноз
            mask_day = (
                pd.to_datetime(fcst_df['dttm_30']).dt.date == d.date()
            )

            current_sum = fcst_df.loc[mask_day, 'forecast'].sum()

            if current_sum <= 0:
                continue

            scale = target_daily_value / current_sum

            fcst_df.loc[mask_day, 'forecast'] *= scale

    return fcst_df.drop(columns=['date'])


def cross_validate(series, holidays, config: Config):
    horizon = config.steps_days
    stride = 7  # шаг окна

    errors = []

    for i in range(0, len(series) - horizon * 2, stride):
        train = series[:len(series) - horizon - i]
        val = series[len(series) - horizon - i: len(series) - i]

        holidays_train = holidays[:len(train)]
        holidays_val = holidays[len(train):len(train)+horizon]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        model.fit(train, future_covariates=holidays_train)
        pred = model.predict(n=horizon, future_covariates=holidays_val)

        y_true = val.values().flatten()
        y_pred = pred.values().flatten()

        mape, wape = calculate_metrics(y_true, y_pred)
        errors.append(wape)

    return np.mean(errors)

def run_forecast(df, holiday_series, get_intraday_profile, aggregate_to_daily, config, output_dir):

    results = []
    metrics_rows = []


    for ts_name in df['ts_name'].unique()[1:3]:
        logger.info(f"Processing TS: {ts_name}")
        ts_df = df[df['ts_name'] == ts_name]

        if len(ts_df) < 48 * 14:
            continue

        profile = get_intraday_profile(ts_df)
        daily_df = aggregate_to_daily(ts_df)

        series_daily = TimeSeries.from_dataframe(daily_df, 'date', 'y', freq='D')

        train_dates = series_daily.time_index

        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        holidays_train_df = pd.DataFrame({
            'date': train_dates
        })

        holidays_train_df['is_holiday'] = holidays_train_df['date'].apply(
            lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0
        )

        holidays_train = TimeSeries.from_dataframe(
            holidays_train_df,
            'date',
            'is_holiday',
            freq='D'
        )


        start_date = series_daily.end_time() + pd.Timedelta(days=1)

        future_dates = pd.date_range(
            start=start_date,
            periods=config.steps_days,
            freq='D'
        )

        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        future_holidays_df = pd.DataFrame({
            'date': future_dates
        })

        future_holidays_df['is_holiday'] = future_holidays_df['date'].apply(
            lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0
        )

        future_holidays = TimeSeries.from_dataframe(
            future_holidays_df,
            'date',
            'is_holiday',
            freq='D'
        )


        fcst_daily, best_params, metrics, study = forecast_daily(series_daily, holidays_train, future_holidays, config, ts_name, output_dir)


        fcst_30 = disaggregate_to_intraday(
            fcst_daily,
            profile,
            series_daily.end_time() + pd.Timedelta(minutes=30),
            config.steps_days
        )

        ny_profile = get_new_year_profile(ts_df)
        fcst_30 = apply_new_year_adjustment(fcst_30, ny_profile)

        fcst_30['ts_name'] = ts_name
        results.append(fcst_30)

        metrics_rows.append({
            "ts_name": ts_name,
            "cv_wape": metrics["cv_wape"],
            "best_wape": metrics["best_wape"],
            "best_params": str(best_params)
        })

    metrics_df = pd.DataFrame(metrics_rows)

    return (
        pd.concat(results, ignore_index=True),
        metrics_df
    )




def save_forecast(df_forecast, output_dir):

    path = os.path.join(output_dir, "forecast.xlsx")

    with pd.ExcelWriter(path) as writer:
        for ts_name in df_forecast['ts_name'].unique():
            df_ts = df_forecast[df_forecast['ts_name'] == ts_name]
            df_ts.to_excel(writer, sheet_name=str(ts_name), index=False)

    logger.info(f"Saved: {path}")


def save_plots(df, forecast_df, output_dir):

    for ts_name in forecast_df['ts_name'].unique():

        df_hist = df[df['ts_name'] == ts_name].copy()
        df_fcst = forecast_df[forecast_df['ts_name'] == ts_name].copy()



        hist_daily = (
            df_hist
            .set_index('dttm_30')
            .resample('D')['y']
            .sum()
            .reset_index()
        )

        fcst_daily = (
            df_fcst
            .set_index('dttm_30')
            .resample('D')['forecast']
            .sum()
            .reset_index()
        )


        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Intraday (30 min)", "Daily aggregation")
        )

        fig.add_trace(
            go.Scatter(
                x=df_hist["dttm_30"],
                y=df_hist["y"],
                name="History",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df_fcst["dttm_30"],
                y=df_fcst["forecast"],
                name="Forecast",
            ),
            row=1, col=1
        )


        fig.add_trace(
            go.Scatter(
                x=hist_daily["dttm_30"],
                y=hist_daily["y"],
                name="History (daily)",
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=fcst_daily["dttm_30"],
                y=fcst_daily["forecast"],
                name="Forecast (daily)",
            ),
            row=2, col=1
        )


        forecast_start = df_fcst["dttm_30"].min()

        fig.add_vline(x=forecast_start, line_dash="dash")

        fig.update_layout(
            height=800,
            title_text=f"Forecast for {ts_name}"
        )


        path = os.path.join(output_dir, f"forecast_{ts_name}.html")
        fig.write_html(path)

        logger.info(f"Saved plot: {path}")


@app.command()
def main(
    file_path: str = typer.Option(..., help="Path to input file"),
    steps_days: int = typer.Option(7),
    use_crossvalidation: bool = typer.Option(False),
    use_params_tuning: bool = typer.Option(False),
):

    if not os.path.exists(file_path):
        typer.echo(f"File not found: {file_path}")
        raise typer.Exit()

    config = Config(
        steps_days=steps_days,
        use_crossvalidation=use_crossvalidation,
        use_params_tuning=use_params_tuning
    )

    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(file_path)

    df, get_intraday_profile, aggregate_to_daily, holiday_series = preprocess(df, config)

    forecast_df, metrics_df = run_forecast(
        df,
        holiday_series,
        get_intraday_profile,
        aggregate_to_daily,
        config,
        output_dir
    )

    save_forecast(forecast_df, output_dir)
    save_plots(df, forecast_df, output_dir)



if __name__ == "__main__":
    app()