from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime, timezone
from model_utils import predict_all_zones, get_actual_for_slot
from map_builder import build_heatmap
from demo_online_model import add_demo_record, get_demo_stats, DEMO_RETRAIN_THRESHOLD
from model_utils import load_resources
from datetime import date

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_page():
    map_html = None
    table_data = None
    error = None

    if request.method == "POST":
        try:
            date_str = request.form.get("date")       # e.g. "2026-07-15"
            time_str = request.form.get("time_slot")   # e.g. "09:00"
            naive_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            target_time = pd.Timestamp(naive_dt, tz='UTC')

            pred_df = predict_all_zones(target_time)
            if pred_df.empty:
                error = "No historical data available to generate a prediction for this slot."
            else:
                map_html = build_heatmap(pred_df, 'predicted_count',
                                          f"Predicted Hotspots — {date_str} {time_str}")
                table_data = pred_df.head(10).to_dict(orient='records')
        except Exception as e:
            error = f"Could not generate prediction: {str(e)}"

    return render_template("predict.html", map_html=map_html, table_data=table_data, error=error)


COMPARE_MIN_DATE = date(2024, 3, 15)
COMPARE_MAX_DATE = date(2024, 4, 7)

@app.route("/compare", methods=["GET", "POST"])
def compare_page():
    pred_map_html = None
    actual_map_html = None
    error = None

    if request.method == "POST":
        try:
            date_str = request.form.get("date")
            time_str = request.form.get("time_slot")

            entered_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if not (COMPARE_MIN_DATE <= entered_date <= COMPARE_MAX_DATE):
                raise ValueError(
                    f"Date must be between {COMPARE_MIN_DATE} and {COMPARE_MAX_DATE} "
                    f"(the test data range)."
                )

            naive_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            target_time = pd.Timestamp(naive_dt, tz='UTC')

            pred_df = predict_all_zones(target_time)
            actual_df = get_actual_for_slot(target_time)

            if actual_df.empty:
                error = "No actual data exists for this slot — pick a date/time from your test period."
            else:
                pred_map_html = build_heatmap(pred_df, 'predicted_count', f"Predicted — {date_str} {time_str}")
                actual_map_html = build_heatmap(actual_df, 'violation_count', f"Actual — {date_str} {time_str}")
        except Exception as e:
            error = f"Could not generate comparison: {str(e)}"

    return render_template("compare.html", pred_map_html=pred_map_html,
                            actual_map_html=actual_map_html, error=error,
                            min_date=COMPARE_MIN_DATE.isoformat(), max_date=COMPARE_MAX_DATE.isoformat())


@app.route("/demo", methods=["GET", "POST"])
def demo_page():
    message = None
    stats = get_demo_stats()

    if request.method == "POST":
        try:
            lat = float(request.form.get("latitude"))
            lon = float(request.form.get("longitude"))
            date_str = request.form.get("date")
            time_str = request.form.get("time_slot")

            # Basic sanity bounds — reject obviously bad/abusive input, no DB/file path exposure
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Latitude/longitude out of valid range.")

            naive_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            target_time = pd.Timestamp(naive_dt, tz='UTC')

            result = add_demo_record(lat, lon, target_time)
            stats = get_demo_stats()
            message = result
        except Exception as e:
            message = f"Invalid submission: {str(e)}"

    return render_template("demo.html", message=message, stats=stats,
                            threshold=DEMO_RETRAIN_THRESHOLD)


if __name__ == "__main__":
    load_resources()  # warm the cache once at startup, not on first user request
    app.run(debug=False, host="0.0.0.0", port=5000)