WITH
client_displays_e_co AS(
  SELECT * 
  FROM lta.bi_client_galactica
  WHERE period = '$PERIOD' AND "day" = '$DAY'
),
define_client AS(
  SELECT DISTINCT client_id, vertical_level_2_name, client_country_code  
  FROM datamart.dim_client
  ORDER BY client_id
),
join_1 AS (
  SELECT * FROM client_displays_e_co LEFT JOIN define_client USING (client_id)   
  WHERE define_client.client_country_code = 'GB' AND define_client.vertical_level_2_name = 'RETAIL'
),
join_2 AS(
  SELECT * 
  FROM join_1  
),
ranking_table AS(
  SELECT client_id, 
  CASE WHEN ranking = 'TIER 1' then 1 ELSE 0 END as is_tier_1 
  FROM datamart.dim_client  
),
join_3 AS(
  SELECT * 
  FROM join_2
  LEFT JOIN ranking_table USING (client_id)
),
spent_in_period_by_day AS (
  SELECT "day", client_id, displays, clicks, revenue_euro
  FROM datamart.fact_client_stats_daily_euro
  WHERE "day" >= TIMESTAMP '$DAY' - '$PERIOD' * INTERVAL '1 day'  AND "day" <= '$DAY'
  ORDER BY client_id 
),
spent_in_period_sum AS (
  SELECT client_id, SUM(displays)AS sum_displays, SUM(clicks) AS sum_clicks, sum(revenue_euro) AS sum_revenue_euro
  FROM spent_in_period_by_day
  GROUP BY client_id
),
join_4 AS (
  SELECT * FROM join_3
  LEFT JOIN spent_in_period_sum USING (client_id)
),
merchant_table AS(
   SELECT most_displayed_client_id, merchant_id, merchant_name
   FROM datamart.dim_merchant
),
join_5 AS (
   SELECT * FROM join_4
   LEFT JOIN merchant_table ON merchant_table.most_displayed_client_id = join_4.client_id
)
SELECT * FROM join_5
ORDER BY client_id;