# AFRISET-site

## Funtions

### get_city_details

- Makes API call to `https://api.airvisual.com/v2/nearest_city?key=YOUR_API_KEY` with header `X_FORWARDED_FOR: USER_IP`
- return response as text and pass it through `json.loads(obj)`

- Returns `obj[0] = city`,  `obj[1] = country`, `obj[2] = aqius`, `obj[3] = weather`, `obj[4] = passed_time`, `obj[5] = w_icon(weather Icon)`, `obj[6] = humidity`
- Passed time is obtained from `datetime.datetime.now().strftime("%H:%M")`

## TODO

- Create a database to store login data for Login page for admin users
- Change tips to interpretation of colors and aqi
- add a survey and contact page
- add a legend
- add a counter based on IP and cookie

### Sensors.Africa IDs

```text
Sensor1: 10676715
Sensor2: 126620191
Sensor3: 10667948
```

### AirBeam PM2.5

```text
Sensor1: AfriSET (1)
```

:::info

- `resample = 'D'` for daily data
- `resample = 'H'` for hourly data

:::

:TODO:

- make graphs to compare with other sensors
- make graphs to compare with other particulate matter data
- make graphs to compare particulate matter data with weather data
- make graphs to compare weather data

- add a calendar plot to show trend during the week
