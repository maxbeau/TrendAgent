import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.services.factors import compute_macro
from app.services.providers.fred import FREDProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_macro():
    logger.info("Starting Macro Factor Test...")
    
    # Initialize Provider
    # Ensure you have FRED_API_KEY in your environment variables or .env file
    fred = FREDProvider()
    
    # Mock weights for testing
    weights = {
        "macro.liquidity_direction": 0.4,
        "macro.credit_spread": 0.2,
        "macro.rate_trend": 0.2,
        "macro.global_demand": 0.2,
    }
    
    try:
        logger.info("Fetching data from FRED and computing scores...")
        result = await compute_macro(fred=fred, factor_weights=weights)
        
        logger.info("-" * 50)
        logger.info(f"Status: {result.status}")
        logger.info(f"Total Score: {result.score}")
        logger.info("-" * 50)
        
        if result.errors:
            logger.error(f"Errors encountered: {result.errors}")
            
        comps = result.components
        logger.info("COMPONENTS DETAILS:")
        logger.info(f"Net Liquidity Latest: {comps.get('net_liquidity_latest')}")
        logger.info(f"Net Liquidity Change: {comps.get('net_liquidity_change')}")
        logger.info(f"Net Liquidity Score:  {comps.get('factor_scores', {}).get('macro.liquidity_direction')}")
        logger.info("-" * 30)
        logger.info(f"Yield Curve (10Y-2Y): {comps.get('yield_curve')}")
        logger.info(f"Yield Curve Score:    {comps.get('yield_curve_score')}")
        logger.info("-" * 30)
        logger.info(f"HY Spread:            {comps.get('high_yield_spread')}")
        logger.info(f"HY Spread Score:      {comps.get('factor_scores', {}).get('macro.credit_spread')}")
        logger.info("-" * 30)
        logger.info(f"Fed Funds Change:     {comps.get('fed_funds_change')}")
        logger.info(f"Rate Trend Score:     {comps.get('factor_scores', {}).get('macro.rate_trend')}")
        logger.info("-" * 30)
        logger.info(f"Ind. Production YoY:  {comps.get('industrial_production_yoy')}")
        logger.info(f"Global Demand Score:  {comps.get('factor_scores', {}).get('macro.global_demand')}")
        
    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_macro())
