from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
from pymongo import MongoClient
import ssl
import os
import requests
import json
from datetime import datetime
import certifi
from bson import ObjectId
import uuid

# Create router instead of FastAPI app
router = APIRouter(prefix="/carbon", tags=["Carbon Calculator"])

# MongoDB connection configuration - Fixed for SSL issues
MONGODB_URL = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
DATABASE_NAME = "rag_db"

# Global variables for MongoDB client and database
client = None
db = None

def debug_environment():
    """Debug function to check MongoDB environment variables"""
    print("=== MongoDB Environment Debug ===")
    print(f"MONGODB_URI: {'Set' if os.getenv('MONGODB_URI') else 'Not set'}")
    print(f"MONGODB_URL: {'Set' if os.getenv('MONGODB_URL') else 'Not set'}")
    
    if os.getenv('MONGODB_URI'):
        uri = os.getenv('MONGODB_URI')
        print(f"MONGODB_URI length: {len(uri)}")
        print(f"MONGODB_URI starts with: {uri[:30]}...")
    
    if os.getenv('MONGODB_URL'):
        url = os.getenv('MONGODB_URL')
        print(f"MONGODB_URL length: {len(url)}")
        print(f"MONGODB_URL starts with: {url[:30]}...")
    
    print(f"All env vars containing 'MONGO': {[k for k in os.environ.keys() if 'MONGO' in k.upper()]}")
    print(f"SSL Certificate Authority file: {certifi.where()}")
    print("================================")

def get_mongodb_client():
    """Initialize MongoDB client with proper SSL configuration for MongoDB Atlas"""
    global client, db
    
    if client is not None:
        return client, db
    
    try:
        if not MONGODB_URL:
            print("WARNING: Neither MONGODB_URI nor MONGODB_URL found in environment variables")
            print("Available env vars:", [k for k in os.environ.keys() if 'MONGO' in k.upper()])
            return None, None
            
        print("Attempting to connect to MongoDB...")
        print(f"Using connection string: {MONGODB_URL[:50]}...") # Only show first 50 chars for security
        
        # Multiple connection strategies to handle SSL issues
        connection_strategies = [
            # Strategy 1: Use certifi CA bundle with TLS 1.2+
            {
                "params": {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "socketTimeoutMS": 30000,
                    "ssl": True,
                    "ssl_cert_reqs": ssl.CERT_REQUIRED,
                    "ssl_ca_certs": certifi.where(),
                    "tlsVersion": "TLSv1.2"
                },
                "description": "SSL with certifi CA bundle and TLS 1.2"
            },
            # Strategy 2: Disable SSL certificate verification (less secure but works)
            {
                "params": {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "socketTimeoutMS": 30000,
                    "ssl": True,
                    "ssl_cert_reqs": ssl.CERT_NONE,
                    "tlsAllowInvalidCertificates": True,
                    "tlsAllowInvalidHostnames": True
                },
                "description": "SSL with disabled certificate verification"
            },
            # Strategy 3: Use connection string SSL parameters only
            {
                "params": {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "socketTimeoutMS": 30000
                },
                "description": "Connection string SSL parameters only"
            },
            # Strategy 4: Alternative SSL context
            {
                "params": {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "socketTimeoutMS": 30000,
                    "ssl": True,
                    "ssl_cert_reqs": ssl.CERT_NONE,
                    "ssl_match_hostname": False
                },
                "description": "SSL with no hostname matching"
            }
        ]
        
        for i, strategy in enumerate(connection_strategies, 1):
            try:
                print(f"Trying connection strategy {i}: {strategy['description']}")
                
                client = MongoClient(MONGODB_URL, **strategy["params"])
                
                # Test the connection
                client.admin.command('ping')
                db = client[DATABASE_NAME]
                
                print(f"MongoDB connected successfully using strategy {i}!")
                print(f"Database: {DATABASE_NAME}")
                
                # Try to list collections to verify connection
                try:
                    collections = db.list_collection_names()
                    print(f"Collections available: {collections}")
                except Exception as e:
                    print(f"Warning: Could not list collections: {e}")
                
                return client, db
                
            except Exception as e:
                print(f"Strategy {i} failed: {str(e)[:200]}...")
                client = None
                db = None
                continue
        
        # If all strategies failed
        print("All connection strategies failed")
        return None, None
        
    except Exception as e:
        print(f"MongoDB connection setup failed: {e}")
        client = None
        db = None
        return None, None

def ensure_db_connection():
    """Ensure database connection is available for endpoints"""
    global client, db
    
    if db is None:
        client, db = get_mongodb_client()
    
    if db is None:
        raise HTTPException(
            status_code=503, 
            detail="Database connection unavailable. Please try again later."
        )
    
    return db

# LLM API Configuration
LLM_API_URL = "https://apillm.mobiloittegroup.com/api/generate"

# Pydantic models for request/response
class TransportData(BaseModel):
    mode: str = Field(..., description="Mode of transport: PetrolCar, DieselCar, Bus, Train, Motorcycle, Walking, Cycle")
    monthly_distance_km: float = Field(..., description="Monthly distance in kilometers")
    description: Optional[str] = Field(None, description="Additional details about vehicle/transport")

class WasteData(BaseModel):
    food_waste: float = Field(0, description="Monthly food waste in kg")
    plastic: float = Field(0, description="Monthly plastic waste in kg")
    paper: float = Field(0, description="Monthly paper waste in kg")
    metal: float = Field(0, description="Monthly metal waste in kg")
    textile: float = Field(0, description="Monthly textile waste in kg")
    electronic: float = Field(0, description="Monthly electronic waste in kg")
    organic_yard: float = Field(0, description="Monthly organic yard waste in kg")

class CarbonCalculationRequest(BaseModel):
    age: int = Field(..., description="Age of the person")
    country: str = Field(..., description="Country name")
    monthly_electricity_kwh: float = Field(..., description="Monthly electricity consumption in kWh")
    transport: list[TransportData] = Field(..., description="List of transport modes and distances")
    eating_habits: str = Field(..., description="Diet type: Pure Vegetarian, Occasional Non-Vegetarian, Regular Non-Vegetarian")
    waste_generation: WasteData = Field(..., description="Monthly waste generation data")
    description: Optional[str] = Field(None, description="Additional details about lifestyle, living situation, work, etc.")

class LLMInsights(BaseModel):
    adjusted_emission_estimate: Optional[float]
    insights: str
    reduction_suggestions: List[str]
    confidence_level: str

class CarbonCalculationResponse(BaseModel):
    total_monthly_emission_kg: float
    breakdown: Dict[str, float]
    details: Dict[str, Any]
    llm_insights: Optional[LLMInsights]
    timestamp: str

class CarbonDataListResponse(BaseModel):
    total_records: int
    data: List[Dict[str, Any]]

class PlantResponse(BaseModel):
    plant_id: int
    common_name: str
    monthly_carbon_absorption_g: int
    plant_type: str
    light_requirement: str
    water_frequency: str
    pot_size_min_inches: int
    soil_type: str
    humidity_preference: str
    temperature_range_f: str
    growth_rate: str
    max_height_inches: int
    care_difficulty: str
    propagation_method: str
    flowering: str
    pet_safe: str
    air_purifying: str

class PlantsListResponse(BaseModel):
    total_plants: int
    plants: List[PlantResponse]

class AddPlantRequest(BaseModel):
    plant_id: int = Field(..., description="ID of the plant from Plants collection")

class PlantOwnedResponse(BaseModel):
    plants_owned_id: str
    plant_id: int
    common_name: str
    monthly_carbon_absorption_g: int
    plant_type: str
    light_requirement: str
    water_frequency: str
    pot_size_min_inches: int
    soil_type: str
    humidity_preference: str
    temperature_range_f: str
    growth_rate: str
    max_height_inches: int
    care_difficulty: str
    propagation_method: str
    flowering: str
    pet_safe: str
    air_purifying: str
    added_at: str

class PlantsOwnedListResponse(BaseModel):
    total_owned_plants: int
    owned_plants: List[PlantOwnedResponse]

class CarbonAbsorptionResponse(BaseModel):
    total_monthly_carbon_absorption_g: int
    total_plants_owned: int

class LatestEmissionResponse(BaseModel):
    total_monthly_emission_kg: float
    timestamp: str
    created_at: str

class SimpleEmissionResponse(BaseModel):
    total_monthly_emission_kg: float

# Helper functions
def get_age_group(age: int) -> str:
    """Convert age to age group format used in database"""
    if age <= 10:
        return "0–10"
    elif age <= 20:
        return "11–20"
    elif age <= 30:
        return "21–30"
    elif age <= 40:
        return "31–40"
    elif age <= 50:
        return "41–50"
    elif age <= 60:
        return "51–60"
    elif age <= 70:
        return "61–70"
    else:
        return "71+"

def calculate_electricity_emission(country: str, kwh: float) -> tuple[float, dict]:
    """Calculate CO2 emission from electricity consumption"""
    try:
        db = ensure_db_connection()
        country_data = db["country-emission"].find_one({"Country": country})
        
        if not country_data:
            raise HTTPException(status_code=404, detail=f"Country '{country}' not found in database")
        
        carbon_intensity = country_data["Carbon intensity of electricity - gCO2/kWh"]
        emission_kg = (kwh * carbon_intensity) / 1000
        
        details = {
            "country": country,
            "kwh_consumed": kwh,
            "carbon_intensity_gCO2_per_kwh": carbon_intensity,
            "emission_kg_co2": emission_kg
        }
        
        return emission_kg, details
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating electricity emission: {str(e)}")

def calculate_diet_emission(age: int, diet_type: str) -> tuple[float, dict]:
    """Calculate CO2 emission from eating habits"""
    try:
        db = ensure_db_connection()
        age_group = get_age_group(age)
        
        diet_data = db["eating_habits"].find_one({
            "Age Group": age_group,
            "Diet Type": diet_type
        })
        
        if not diet_data:
            raise HTTPException(status_code=404, 
                              detail=f"Diet data not found for age group '{age_group}' and diet type '{diet_type}'")
        
        emission_kg = diet_data["monthly Carbon Emission (kg CO₂e)"]
        
        details = {
            "age": age,
            "age_group": age_group,
            "diet_type": diet_type,
            "emission_kg_co2": emission_kg
        }
        
        return emission_kg, details
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating diet emission: {str(e)}")

def calculate_transport_emission(transport_list: list[TransportData]) -> tuple[float, dict]:
    """Calculate CO2 emission from transportation"""
    try:
        db = ensure_db_connection()
        transport_data = db["extras"].find_one({"Title": "Transport Emission"})
        
        if not transport_data:
            raise HTTPException(status_code=404, detail="Transport emission data not found")
        
        total_emission = 0
        transport_details = []
        
        for transport in transport_list:
            if transport.mode not in transport_data:
                raise HTTPException(status_code=400, 
                                  detail=f"Transport mode '{transport.mode}' not supported")
            
            emission_per_km = transport_data[transport.mode]
            transport_emission = transport.monthly_distance_km * emission_per_km
            total_emission += transport_emission
            
            transport_details.append({
                "mode": transport.mode,
                "monthly_distance_km": transport.monthly_distance_km,
                "emission_per_km": emission_per_km,
                "total_emission_kg": transport_emission,
                "description": transport.description
            })
        
        details = {
            "total_emission_kg": total_emission,
            "transport_breakdown": transport_details
        }
        
        return total_emission, details
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating transport emission: {str(e)}")

def calculate_waste_emission(waste: WasteData) -> tuple[float, dict]:
    """Calculate CO2 emission from waste generation"""
    try:
        db = ensure_db_connection()
        waste_data = db["extras"].find_one({"Title": "Waste"})
        
        if not waste_data:
            raise HTTPException(status_code=404, detail="Waste emission data not found")
        
        waste_mapping = {
            "food_waste": "Food waste",
            "plastic": "Plastic",
            "paper": "Paper",
            "metal": "Metal",
            "textile": "Textile",
            "electronic": "Electronic",
            "organic_yard": "Organic yard"
        }
        
        total_emission = 0
        waste_details = []
        
        for waste_type, db_key in waste_mapping.items():
            waste_amount = getattr(waste, waste_type)
            
            if waste_amount > 0:
                emission_per_kg = waste_data[db_key]
                waste_emission = waste_amount * emission_per_kg
                total_emission += waste_emission
                
                waste_details.append({
                    "waste_type": waste_type,
                    "amount_kg": waste_amount,
                    "emission_per_kg": emission_per_kg,
                    "total_emission_kg": waste_emission
                })
        
        details = {
            "total_emission_kg": total_emission,
            "waste_breakdown": waste_details
        }
        
        return total_emission, details
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating waste emission: {str(e)}")

def get_llm_insights(calculation_data: dict, user_description: str) -> Optional[LLMInsights]:
    """Get insights and suggestions from LLM based on carbon calculation and user description"""
    try:
        # Create a comprehensive prompt for the LLM
        prompt = f"""
You are a carbon footprint expert and environmental advisor. Based on the following carbon emission calculation and user description, provide insights and suggestions.

CARBON EMISSION BREAKDOWN:
- Total Monthly Emission: {calculation_data['total_emission']} kg CO₂
- Electricity: {calculation_data['breakdown']['electricity_kg_co2']} kg CO₂
- Diet: {calculation_data['breakdown']['diet_kg_co2']} kg CO₂  
- Transport: {calculation_data['breakdown']['transport_kg_co2']} kg CO₂
- Waste: {calculation_data['breakdown']['waste_kg_co2']} kg CO₂

USER DETAILS:
- Age: {calculation_data['age']}
- Country: {calculation_data['country']}
- Diet Type: {calculation_data['diet_type']}
- Monthly Electricity: {calculation_data['electricity_kwh']} kWh
- Transport Details: {json.dumps(calculation_data['transport_details'], indent=2)}
- Additional Description: {user_description or 'No additional details provided'}

Please provide:
1. An adjusted emission estimate if the user description suggests factors not captured in the standard calculation
2. Key insights about their carbon footprint
3. 5-7 practical, personalized suggestions to reduce their carbon emissions
4. Your confidence level in the analysis (High/Medium/Low)

Format your response as JSON with the following structure:
{{
    "adjusted_emission_estimate": <number or null if no adjustment needed>,
    "insights": "<detailed insights about their carbon footprint>",
    "reduction_suggestions": [
        "<suggestion 1>",
        "<suggestion 2>",
        "<suggestion 3>",
        "<suggestion 4>",
        "<suggestion 5>"
    ],
    "confidence_level": "<High/Medium/Low>"
}}
"""

        # Make API call to LLM
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            llm_response = response.json()
            llm_text = llm_response.get("response", "").strip()
            
            # Try to parse JSON from LLM response
            try:
                # Sometimes LLM might wrap JSON in code blocks, so we clean it
                if "```json" in llm_text:
                    llm_text = llm_text.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_text:
                    llm_text = llm_text.split("```")[1].strip()
                
                llm_data = json.loads(llm_text)
                
                return LLMInsights(
                    adjusted_emission_estimate=llm_data.get("adjusted_emission_estimate"),
                    insights=llm_data.get("insights", ""),
                    reduction_suggestions=llm_data.get("reduction_suggestions", []),
                    confidence_level=llm_data.get("confidence_level", "Medium")
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, create a basic response
                return LLMInsights(
                    adjusted_emission_estimate=None,
                    insights=llm_text[:500] + "..." if len(llm_text) > 500 else llm_text,
                    reduction_suggestions=["Reduce energy consumption", "Use public transport", "Reduce waste generation"],
                    confidence_level="Low"
                )
        else:
            print(f"LLM API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting LLM insights: {str(e)}")
        return None

def save_carbon_data(request_data: dict, response_data: dict, timestamp: str) -> str:
    """Save carbon calculation data to MongoDB with timestamp as unique identifier"""
    try:
        db = ensure_db_connection()
        document = {
            "timestamp": timestamp,
            "request_data": request_data,
            "response_data": response_data,
            "created_at": datetime.now().isoformat()
        }
        
        result = db["carbon_data"].insert_one(document)
        return timestamp
        
    except Exception as e:
        print(f"Error saving carbon data: {str(e)}")
        return ""

# Initialize MongoDB connection when module loads
print("Initializing MongoDB connection...")
debug_environment()  # Debug environment variables
get_mongodb_client()

# API Endpoints
@router.get("/health")
async def carbon_health_check():
    """Health check endpoint to verify database connection"""
    try:
        db = ensure_db_connection()
        # Test database connection
        db.list_collection_names()
        collections = db.list_collection_names()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "timestamp": datetime.now().isoformat(),
            "database_name": DATABASE_NAME,
            "collections_count": len(collections),
            "collections": collections[:5]  # Show first 5 collections
        }
    except HTTPException as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected", 
            "timestamp": datetime.now().isoformat(),
            "error": str(e.detail)
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected", 
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.post("/calculate-emission", response_model=CarbonCalculationResponse)
async def calculate_carbon_emission(request: CarbonCalculationRequest):
    """Calculate monthly carbon emission based on various factors with LLM insights"""
    try:
        # Generate timestamp for this calculation
        calculation_timestamp = datetime.now().isoformat()
        
        # Calculate emissions from different sources
        electricity_emission, electricity_details = calculate_electricity_emission(
            request.country, request.monthly_electricity_kwh
        )
        
        diet_emission, diet_details = calculate_diet_emission(
            request.age, request.eating_habits
        )
        
        transport_emission, transport_details = calculate_transport_emission(
            request.transport
        )
        
        waste_emission, waste_details = calculate_waste_emission(
            request.waste_generation
        )
        
        # Calculate total emission
        total_emission = electricity_emission + diet_emission + transport_emission + waste_emission
        
        # Prepare calculation data for LLM
        calculation_data = {
            "total_emission": round(total_emission, 2),
            "breakdown": {
                "electricity_kg_co2": round(electricity_emission, 2),
                "diet_kg_co2": round(diet_emission, 2),
                "transport_kg_co2": round(transport_emission, 2),
                "waste_kg_co2": round(waste_emission, 2)
            },
            "age": request.age,
            "country": request.country,
            "diet_type": request.eating_habits,
            "electricity_kwh": request.monthly_electricity_kwh,
            "transport_details": [t.dict() for t in request.transport]
        }
        
        # Get LLM insights
        llm_insights = None
        if request.description:
            llm_insights = get_llm_insights(calculation_data, request.description)
        
        # Prepare response
        response_data = {
            "total_monthly_emission_kg": round(total_emission, 2),
            "breakdown": calculation_data["breakdown"],
            "details": {
                "electricity": electricity_details,
                "diet": diet_details,
                "transport": transport_details,
                "waste": waste_details,
                "user_description": request.description,
                "calculation_timestamp": calculation_timestamp
            },
            "llm_insights": llm_insights.dict() if llm_insights else None,
            "timestamp": calculation_timestamp
        }
        
        # Save to database
        save_carbon_data(request.dict(), response_data, calculation_timestamp)
        
        response = CarbonCalculationResponse(**response_data)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating carbon emission: {str(e)}")

@router.get("/calculations", response_model=CarbonDataListResponse)
async def get_carbon_calculations(
    limit: int = 10,
    offset: int = 0,
    sort_order: str = "desc"
):
    """Get list of carbon calculations stored in database with timestamps"""
    try:
        db = ensure_db_connection()
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        
        # Get total count
        total_count = db["carbon_data"].count_documents({})
        
        # Get paginated results with timestamp field included
        cursor = db["carbon_data"].find(
            {},
            {
                "_id": 1,
                "timestamp": 1,
                "response_data.total_monthly_emission_kg": 1,
                "response_data.breakdown": 1,
                "request_data.country": 1,
                "request_data.age": 1,
                "created_at": 1
            }
        ).sort("timestamp", sort_direction).skip(offset).limit(limit)
        
        results = []
        for doc in cursor:
            # Convert ObjectId to string and structure the response
            result_item = {
                "_id": str(doc["_id"]),
                "timestamp": doc.get("timestamp"),
                "total_monthly_emission_kg": doc.get("response_data", {}).get("total_monthly_emission_kg"),
                "breakdown": doc.get("response_data", {}).get("breakdown"),
                "country": doc.get("request_data", {}).get("country"),
                "age": doc.get("request_data", {}).get("age"),
                "created_at": doc.get("created_at")
            }
            results.append(result_item)
        
        return CarbonDataListResponse(
            total_records=total_count,
            data=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching carbon calculations: {str(e)}")

@router.get("/calculations/{timestamp}")
async def get_carbon_calculation_by_timestamp(timestamp: str):
    """Get specific carbon calculation by timestamp"""
    try:
        db = ensure_db_connection()
        
        # Decode URL-encoded timestamp
        from urllib.parse import unquote
        timestamp = unquote(timestamp)
        
        # First try exact match
        result = db["carbon_data"].find_one({"timestamp": timestamp})
        
        # If not found, try to find by truncated timestamp (handle precision differences)
        if not result:
            # Try to match with different precision levels
            timestamp_base = timestamp.split('.')[0]  # Remove microseconds
            
            # Find documents where timestamp starts with the base timestamp
            result = db["carbon_data"].find_one({
                "timestamp": {"$regex": f"^{timestamp_base}"}
            })
        
        # If still not found, try matching by created_at field as fallback
        if not result:
            result = db["carbon_data"].find_one({"created_at": timestamp})
        
        # Additional fallback - try with timezone variations
        if not result:
            for suffix in ["Z", "+00:00", ".000Z", ".000000Z"]:
                result = db["carbon_data"].find_one({"timestamp": timestamp + suffix})
                if result:
                    break
        
        if not result:
            raise HTTPException(status_code=404, detail="Calculation not found")
        
        result["_id"] = str(result["_id"])
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching calculation: {str(e)}")

@router.get("/supported-countries")
async def get_supported_countries():
    """Get list of supported countries"""
    try:
        db = ensure_db_connection()
        countries = db["country-emission"].distinct("Country")
        return {"countries": sorted(countries)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching countries: {str(e)}")

@router.get("/transport-modes")
async def get_transport_modes():
    """Get available transport modes and their emission factors"""
    try:
        db = ensure_db_connection()
        transport_data = db["extras"].find_one({"Title": "Transport Emission"})
        if not transport_data:
            raise HTTPException(status_code=404, detail="Transport data not found")
        
        transport_modes = {k: v for k, v in transport_data.items() if k not in ["_id", "Title"]}
        return {"transport_modes": transport_modes, "unit": "kg CO2 per km"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transport modes: {str(e)}")

@router.get("/diet-types")
async def get_diet_types():
    """Get available diet types"""
    try:
        db = ensure_db_connection()
        diet_types = db["eating_habits"].distinct("Diet Type")
        age_groups = db["eating_habits"].distinct("Age Group")
        return {
            "diet_types": sorted(diet_types),
            "age_groups": sorted(age_groups)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching diet types: {str(e)}")

@router.delete("/calculations/{timestamp}")
async def delete_carbon_calculation(timestamp: str):
    """Delete a specific carbon calculation by timestamp"""
    try:
        db = ensure_db_connection()
        
        # First try exact match
        result = db["carbon_data"].delete_one({"timestamp": timestamp})
        
        # If not found, try to find by truncated timestamp (handle precision differences)
        if result.deleted_count == 0:
            timestamp_base = timestamp.split('.')[0]  # Remove microseconds
            
            # Find and delete document where timestamp starts with the base timestamp
            result = db["carbon_data"].delete_one({
                "timestamp": {"$regex": f"^{timestamp_base}"}
            })
        
        # If still not found, try matching by created_at field as fallback
        if result.deleted_count == 0:
            result = db["carbon_data"].delete_one({"created_at": timestamp})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Calculation not found")
        
        return {"message": "Calculation deleted successfully", "timestamp": timestamp}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting calculation: {str(e)}")

@router.get("/timestamps")
async def get_all_timestamps():
    """Get all timestamps from carbon calculations"""
    try:
        db = ensure_db_connection()
        timestamps = db["carbon_data"].find({}, {"timestamp": 1, "_id": 0}).sort("timestamp", -1)
        timestamp_list = [doc["timestamp"] for doc in timestamps]
        
        return {
            "total_calculations": len(timestamp_list),
            "timestamps": timestamp_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching timestamps: {str(e)}")

# Debug endpoint to check environment variables
@router.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables (use only in development)"""
    return {
        "mongodb_uri_set": bool(os.getenv("MONGODB_URI")),
        "mongodb_url_set": bool(os.getenv("MONGODB_URL")),
        "mongodb_uri_length": len(os.getenv("MONGODB_URI", "")),
        "mongodb_url_length": len(os.getenv("MONGODB_URL", "")),
        "mongodb_vars": [k for k in os.environ.keys() if 'MONGO' in k.upper()],
        "database_name": DATABASE_NAME,
        "client_status": "Connected" if client is not None else "Disconnected",
        "db_status": "Available" if db is not None else "Unavailable"
    }


@router.get("/plants", response_model=PlantsListResponse)
async def get_all_plants():
    """Get all plants from the Plants collection"""
    try:
        db = ensure_db_connection()
        
        # Get all plants from the Plants collection
        plants_cursor = db["Plants"].find({}, {"_id": 0})  # Exclude MongoDB _id field
        plants_list = list(plants_cursor)
        
        # Convert to response format
        plants_response = []
        for plant in plants_list:
            plant_response = PlantResponse(**plant)
            plants_response.append(plant_response)
        
        return PlantsListResponse(
            total_plants=len(plants_response),
            plants=plants_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching plants: {str(e)}")

@router.post("/plants/add-to-owned")
async def add_plant_to_owned(request: AddPlantRequest):
    """Add a plant from Plants collection to plants_owned collection"""
    try:
        db = ensure_db_connection()
        
        # First, find the plant in the Plants collection
        plant = db["Plants"].find_one({"plant_id": request.plant_id})
        
        if not plant:
            raise HTTPException(
                status_code=404, 
                detail=f"Plant with plant_id {request.plant_id} not found in Plants collection"
            )
        
        # Generate a unique plants_owned_id
        plants_owned_id = str(uuid.uuid4())
        
        # Create the owned plant document
        owned_plant = {
            "plants_owned_id": plants_owned_id,
            "plant_id": plant["plant_id"],
            "common_name": plant["common_name"],
            "monthly_carbon_absorption_g": plant["monthly_carbon_absorption_g"],
            "plant_type": plant["plant_type"],
            "light_requirement": plant["light_requirement"],
            "water_frequency": plant["water_frequency"],
            "pot_size_min_inches": plant["pot_size_min_inches"],
            "soil_type": plant["soil_type"],
            "humidity_preference": plant["humidity_preference"],
            "temperature_range_f": plant["temperature_range_f"],
            "growth_rate": plant["growth_rate"],
            "max_height_inches": plant["max_height_inches"],
            "care_difficulty": plant["care_difficulty"],
            "propagation_method": plant["propagation_method"],
            "flowering": plant["flowering"],
            "pet_safe": plant["pet_safe"],
            "air_purifying": plant["air_purifying"],
            "added_at": datetime.now().isoformat()
        }
        
        # Insert into plants_owned collection
        result = db["plants_owned"].insert_one(owned_plant)
        
        if result.inserted_id:
            return {
                "message": "Plant added to owned plants successfully",
                "plants_owned_id": plants_owned_id,
                "plant_id": request.plant_id,
                "common_name": plant["common_name"],
                "added_at": owned_plant["added_at"]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add plant to owned plants")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding plant to owned: {str(e)}")

@router.get("/plants/owned", response_model=PlantsOwnedListResponse)
async def get_owned_plants():
    """Get all plants from the plants_owned collection"""
    try:
        db = ensure_db_connection()
        
        # Get all owned plants from the plants_owned collection
        owned_plants_cursor = db["plants_owned"].find({}, {"_id": 0})  # Exclude MongoDB _id field
        owned_plants_list = list(owned_plants_cursor)
        
        # Convert to response format
        owned_plants_response = []
        for owned_plant in owned_plants_list:
            plant_owned_response = PlantOwnedResponse(**owned_plant)
            owned_plants_response.append(plant_owned_response)
        
        return PlantsOwnedListResponse(
            total_owned_plants=len(owned_plants_response),
            owned_plants=owned_plants_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching owned plants: {str(e)}")

@router.delete("/plants/owned/{plants_owned_id}")
async def delete_owned_plant(plants_owned_id: str):
    """Delete a plant from plants_owned collection using plants_owned_id"""
    try:
        db = ensure_db_connection()
        
        # Find and delete the owned plant by plants_owned_id
        result = db["plants_owned"].delete_one({"plants_owned_id": plants_owned_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Owned plant with plants_owned_id '{plants_owned_id}' not found"
            )
        
        return {
            "message": "Owned plant deleted successfully",
            "plants_owned_id": plants_owned_id,
            "deleted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting owned plant: {str(e)}")

@router.get("/plants/{plant_id}")
async def get_plant_by_id(plant_id: int):
    """Get a specific plant by plant_id from Plants collection"""
    try:
        db = ensure_db_connection()
        
        # Find the plant by plant_id
        plant = db["Plants"].find_one({"plant_id": plant_id}, {"_id": 0})
        
        if not plant:
            raise HTTPException(
                status_code=404, 
                detail=f"Plant with plant_id {plant_id} not found"
            )
        
        return PlantResponse(**plant)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching plant: {str(e)}")

@router.get("/plants/owned/{plants_owned_id}")
async def get_owned_plant_by_id(plants_owned_id: str):
    """Get a specific owned plant by plants_owned_id"""
    try:
        db = ensure_db_connection()
        
        # Find the owned plant by plants_owned_id
        owned_plant = db["plants_owned"].find_one({"plants_owned_id": plants_owned_id}, {"_id": 0})
        
        if not owned_plant:
            raise HTTPException(
                status_code=404, 
                detail=f"Owned plant with plants_owned_id '{plants_owned_id}' not found"
            )
        
        return PlantOwnedResponse(**owned_plant)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching owned plant: {str(e)}")

@router.get("/plants/owned/by-plant-id/{plant_id}")
async def get_owned_plants_by_plant_id(plant_id: int):
    """Get all owned plants with a specific plant_id"""
    try:
        db = ensure_db_connection()
        
        # Find all owned plants with the specified plant_id
        owned_plants_cursor = db["plants_owned"].find({"plant_id": plant_id}, {"_id": 0})
        owned_plants_list = list(owned_plants_cursor)
        
        if not owned_plants_list:
            return {
                "message": f"No owned plants found with plant_id {plant_id}",
                "plant_id": plant_id,
                "total_owned": 0,
                "owned_plants": []
            }
        
        # Convert to response format
        owned_plants_response = []
        for owned_plant in owned_plants_list:
            plant_owned_response = PlantOwnedResponse(**owned_plant)
            owned_plants_response.append(plant_owned_response)
        
        return {
            "plant_id": plant_id,
            "total_owned": len(owned_plants_response),
            "owned_plants": owned_plants_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching owned plants by plant_id: {str(e)}")
    
@router.get("/plants-owned/carbon-absorption", response_model=CarbonAbsorptionResponse)
async def get_total_carbon_absorption():
    """Calculate total monthly carbon absorption for all owned plants"""
    try:
        db = ensure_db_connection()
        
        # Use MongoDB aggregation pipeline to sum the monthly_carbon_absorption_g field
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_carbon_absorption": {"$sum": "$monthly_carbon_absorption_g"},
                    "plant_count": {"$sum": 1}
                }
            }
        ]
        
        result = list(db["plants_owned"].aggregate(pipeline))
        
        if result:
            return CarbonAbsorptionResponse(
                total_monthly_carbon_absorption_g=result[0]["total_carbon_absorption"],
                total_plants_owned=result[0]["plant_count"]
            )
        else:
            # No plants found in collection
            return CarbonAbsorptionResponse(
                total_monthly_carbon_absorption_g=0,
                total_plants_owned=0
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating carbon absorption: {str(e)}")


@router.get("/carbon-data/latest-emission", response_model=LatestEmissionResponse)
async def get_latest_carbon_emission():
    """Get the total monthly emission from the latest added carbon data record"""
    try:
        db = ensure_db_connection()
        
        # Find the latest document based on created_at field (descending order)
        latest_carbon_data = db["carbon_data"].find_one(
            {},
            {"response_data.total_monthly_emission_kg": 1, 
             "timestamp": 1, 
             "created_at": 1, 
             "_id": 0}
        ).sort("created_at", -1)
        
        if not latest_carbon_data:
            raise HTTPException(status_code=404, detail="No carbon data found")
        
        # Extract the total_monthly_emission_kg from response_data
        total_emission = latest_carbon_data["response_data"]["total_monthly_emission_kg"]
        
        return LatestEmissionResponse(
            total_monthly_emission_kg=total_emission,
            timestamp=latest_carbon_data["timestamp"],
            created_at=latest_carbon_data["created_at"]
        )
        
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing expected field in carbon data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching latest carbon emission: {str(e)}")





# Alternative implementation using aggregation pipeline (more efficient for large datasets)
@router.get("/carbon-data/latest-emission-alt", response_model=LatestEmissionResponse)
async def get_latest_carbon_emission_alt():
    """Get the total monthly emission from the latest added carbon data record using aggregation"""
    try:
        db = ensure_db_connection()
        
        # Use aggregation pipeline to get the latest document
        pipeline = [
            {
                "$sort": {"created_at": -1}  # Sort by created_at descending
            },
            {
                "$limit": 1  # Get only the first (latest) document
            },
            {
                "$project": {
                    "_id": 0,
                    "total_monthly_emission_kg": "$response_data.total_monthly_emission_kg",
                    "timestamp": 1,
                    "created_at": 1
                }
            }
        ]
        
        result = list(db["carbon_data"].aggregate(pipeline))
        
        if not result:
            raise HTTPException(status_code=404, detail="No carbon data found")
        
        latest_data = result[0]
        
        # Check if the required field exists
        if "total_monthly_emission_kg" not in latest_data or latest_data["total_monthly_emission_kg"] is None:
            raise HTTPException(status_code=500, detail="Missing total_monthly_emission_kg in latest carbon data")
        
        return LatestEmissionResponse(
            total_monthly_emission_kg=latest_data["total_monthly_emission_kg"],
            timestamp=latest_data["timestamp"],
            created_at=latest_data["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # More detailed error logging
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error fetching latest carbon emission: {str(e)}")