import time
import datetime
def trde(start_date, end_date, investment, comapany_list, stock_list):
    cur_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    res = end_date - cur_date
    wks = res.days//7
    days = res.days - (wks*2)
    if len(comapany_list) > 1:
        returns = get_stock_value(days, investment)
    else:
        returns = get_stock_value_multi(days,investment)
    time.sleep(30 *days)
    return returns