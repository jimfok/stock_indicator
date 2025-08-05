import screen_manager as screen

# screener 1
SCAN_TYPE = 'FTD'			# Scan marks for: K1 or FTD or or PBB or BuyRating or vol_up or mm_stage2 or BOTH  ( K1 sell mark has bug, BOTH=K1+FTD )
PERIOD = 1					# scan for marks within n days
INTERVAL = '1d'				# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
PRICE_ABOVE = 12			# included PRICE_ABOVE
VOLUMN_ABOVE = 0			# included VOLUMN_ABOVE, suggest 0 or 500000

DEBUG = False				# debug flag
DEBUG_SYMBOL = 'SMCI'		# debug symbol

parameter = [SCAN_TYPE,PERIOD,INTERVAL,PRICE_ABOVE,VOLUMN_ABOVE]
debug_parameter = [DEBUG,DEBUG_SYMBOL]
screen.screen(parameter,debug_parameter)

# screener 2
# SCAN_TYPE = 'mm_stage2'		# Scan marks for: K1 or FTD or or PBB or BuyRating or vol_up or mm_stage2 or BOTH  ( K1 sell mark has bug, BOTH=K1+FTD )
# PERIOD = 10					# scan for marks within n days
# INTERVAL = '1d'				# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
# PRICE_ABOVE = 12			# included PRICE_ABOVE
# VOLUMN_ABOVE = 0			# included VOLUMN_ABOVE, suggest 0 or 500000

# DEBUG = False				# debug flag
# DEBUG_SYMBOL = 'DHI'		# debug symbol

# parameter = [SCAN_TYPE,PERIOD,INTERVAL,PRICE_ABOVE,VOLUMN_ABOVE]
# debug_parameter = [DEBUG,DEBUG_SYMBOL]
# screen.screen(parameter,debug_parameter)