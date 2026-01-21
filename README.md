Weather Forecasting Web Application

This project is a Django-based web application that provides weather forecasting for a given city. It displays temperature, humidity, and rain prediction, and visualizes forecast data using an interactive line chart powered by Chart.js.

The system demonstrates a complete web application workflow:

User Input → Django View Processing → Data Preparation → Template Rendering → JavaScript Visualization

Project Structure:

WeatherProject

forecast

* static

  * css

    * style.css
  * js

    * test.js
* templates

  * weather.html
* views.py
* urls.py
* models.py

WeatherProject

* settings.py
* urls.py
* wsgi.py
* asgi.py

db.sqlite3
manage.py
README.md

Setup:

Create a virtual environment (optional):

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install django

Run the Application:

Start the Django development server:

python manage.py runserver

Open the browser and visit:

[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
or
[http://127.0.0.1:8000/weather/](http://127.0.0.1:8000/weather/)

(depending on URL configuration)

Application Flow:

User enters city name
↓
URL routed via urls.py
↓
weather_view() in views.py
↓
Weather data processed
↓
Context sent to weather.html
↓
Page rendered in browser
↓
test.js loads and generates chart using Chart.js

Highlights:

This project demonstrates a complete Django web workflow, integrates backend logic with frontend visualization, dynamically renders weather data, and uses JavaScript with Chart.js for real-time graph generation.
It showcases practical experience in full-stack development using Django, HTML, CSS, and JavaScript, and provides a clean foundation for extending into API-based real-time weather systems.
