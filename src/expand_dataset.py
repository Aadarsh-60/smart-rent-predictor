import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# Define State -> Cities with Tier and realistic base price per sq ft
STATE_CITY_CONFIG = {
    'Maharashtra': {
        'Tier 1': {
            'Mumbai': {
                'localities': {
                    'Prime / South Mumbai': ['Bandra West', 'Worli', 'Juhu', 'Colaba', 'Lower Parel'],
                    'Mid-City Suburbs': ['Andheri West', 'Powai', 'Goregaon East', 'Chembur', 'Malad West'],
                    'Affordable Suburbs': ['Thane West', 'Navi Mumbai Vashi', 'Kandivali East', 'Borivali West', 'Mira Road']
                },
                'sqft_rate': {'Prime / South Mumbai': (60, 95), 'Mid-City Suburbs': (35, 52), 'Affordable Suburbs': (20, 30)}
            },
            'Pune': {
                'localities': {
                    'Prime / IT Hub': ['Koregaon Park', 'Kalyani Nagar', 'Baner', 'Hinjewadi Phase 1'],
                    'Standard Residential': ['Kothrud', 'Aundh', 'Wakad', 'Viman Nagar'],
                    'Affordable Suburbs': ['Hadapsar', 'Pimpri', 'Chinchwad', 'Kharadi']
                },
                'sqft_rate': {'Prime / IT Hub': (22, 34), 'Standard Residential': (16, 24), 'Affordable Suburbs': (12, 17)}
            }
        },
        'Tier 2': {
            'Nagpur': {
                'localities': {'Prime / Central': ['Dharampeth', 'Ramdaspeth'], 'Standard Residential': ['Civil Lines', 'Manish Nagar'], 'Affordable Suburbs': ['Trimurti Nagar', 'Besa']},
                'sqft_rate': {'Prime / Central': (15, 22), 'Standard Residential': (11, 16), 'Affordable Suburbs': (8, 12)}
            },
            'Nashik': {
                'localities': {'Prime / Central': ['College Road', 'Gangapur Road'], 'Standard Residential': ['Indira Nagar', 'Panchavati'], 'Affordable Suburbs': ['Ambad', 'Nashik Road']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (7, 10)}
            }
        }
    },
    'Karnataka': {
        'Tier 1': {
            'Bangalore': {
                'localities': {
                    'Prime / IT Hub': ['Indiranagar', 'Koramangala', 'HSR Layout', 'Whitefield', 'Bellandur'],
                    'Standard Residential': ['Jayanagar', 'JP Nagar', 'Malleshwaram', 'Hebbal'],
                    'Affordable Suburbs': ['Electronic City', 'Yelahanka', 'Sarjapur Road', 'Kengeri']
                },
                'sqft_rate': {'Prime / IT Hub': (26, 40), 'Standard Residential': (18, 26), 'Affordable Suburbs': (12, 18)}
            }
        },
        'Tier 2': {
            'Mysore': {
                'localities': {'Prime / Central': ['Gokulam', 'Jayalakshmipuram'], 'Standard Residential': ['Kuvempunagar', 'Saraswathipuram'], 'Affordable Suburbs': ['Vijayanagar', 'Hebbal 2nd Stage']},
                'sqft_rate': {'Prime / Central': (14, 20), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 11)}
            },
            'Mangalore': {
                'localities': {'Prime / Central': ['Kadri', 'Bejai'], 'Standard Residential': ['Kankanady', 'Urwa'], 'Affordable Suburbs': ['Surathkal', 'Kavoor']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (7, 10)}
            }
        }
    },
    'Delhi NCR': {
        'Tier 1': {
            'New Delhi': {
                'localities': {
                    'Prime / South Delhi': ['Greater Kailash', 'Hauz Khas', 'Vasant Kunj', 'Saket'],
                    'Standard Residential': ['Dwarka Sector 10', 'Rohini Sector 13', 'Lajpat Nagar', 'Janakpuri'],
                    'Affordable Suburbs': ['Uttam Nagar', 'Burari', 'Mayur Vihar Phase 3', 'Okhla']
                },
                'sqft_rate': {'Prime / South Delhi': (32, 50), 'Standard Residential': (20, 30), 'Affordable Suburbs': (13, 19)}
            },
            'Gurgaon': {
                'localities': {
                    'Prime / IT Hub': ['DLF Phase 5', 'Golf Course Road', 'Cyber City', 'Sohna Road'],
                    'Standard Residential': ['Sector 47', 'Sector 56', 'Palam Vihar'],
                    'Affordable Suburbs': ['Sector 92', 'New Gurgaon Sector 82', 'Sohna']
                },
                'sqft_rate': {'Prime / IT Hub': (28, 45), 'Standard Residential': (18, 27), 'Affordable Suburbs': (12, 17)}
            },
            'Noida': {
                'localities': {
                    'Prime / Expressway': ['Sector 137', 'Sector 150', 'Sector 50', 'Sector 62'],
                    'Standard Residential': ['Sector 76', 'Sector 78', 'Sector 18 Atta'],
                    'Affordable Suburbs': ['Greater Noida West', 'Sector 73', 'Noida Extension']
                },
                'sqft_rate': {'Prime / Expressway': (20, 32), 'Standard Residential': (14, 21), 'Affordable Suburbs': (9, 14)}
            }
        }
    },
    'Telangana': {
        'Tier 1': {
            'Hyderabad': {
                'localities': {
                    'Prime / IT Corridor': ['Gachibowli', 'Hitec City', 'Jubilee Hills', 'Madhapur', 'Banjara Hills'],
                    'Standard Residential': ['Kukatpally', 'Kondapur', 'Manikonda', 'Miyapur'],
                    'Affordable Suburbs': ['Uppal', 'LB Nagar', 'Kompally', 'Bachupally']
                },
                'sqft_rate': {'Prime / IT Corridor': (24, 38), 'Standard Residential': (16, 24), 'Affordable Suburbs': (11, 16)}
            }
        },
        'Tier 2': {
            'Warangal': {
                'localities': {'Prime / Central': ['Hanamkonda', 'Subedari'], 'Standard Residential': ['Kazipet', 'Nayeem Nagar'], 'Affordable Suburbs': ['Hunter Road', 'Waddepally']},
                'sqft_rate': {'Prime / Central': (12, 17), 'Standard Residential': (8, 13), 'Affordable Suburbs': (6, 9)}
            }
        }
    },
    'Tamil Nadu': {
        'Tier 1': {
            'Chennai': {
                'localities': {
                    'Prime / Central': ['Adyar', 'Anna Nagar', 'Besant Nagar', 'T Nagar'],
                    'Standard Residential': ['Velachery', 'OMR Thoraipakkam', 'Mylapore', 'Sholinganallur'],
                    'Affordable Suburbs': ['Tambaram', 'Perambur', 'Chromepet', 'Medavakkam']
                },
                'sqft_rate': {'Prime / Central': (24, 38), 'Standard Residential': (16, 24), 'Affordable Suburbs': (11, 16)}
            }
        },
        'Tier 2': {
            'Coimbatore': {
                'localities': {'Prime / Central': ['RS Puram', 'Race Course'], 'Standard Residential': ['Gandhipuram', 'Peelamedu', 'Saravanampatti'], 'Affordable Suburbs': ['Singanallur', 'Vadavalli']},
                'sqft_rate': {'Prime / Central': (15, 22), 'Standard Residential': (10, 16), 'Affordable Suburbs': (7, 11)}
            },
            'Madurai': {
                'localities': {'Prime / Central': ['KK Nagar', 'Anna Nagar'], 'Standard Residential': ['SS Colony', 'Tallakulam'], 'Affordable Suburbs': ['Thirunagar', 'Sellur']},
                'sqft_rate': {'Prime / Central': (12, 17), 'Standard Residential': (8, 13), 'Affordable Suburbs': (6, 9)}
            }
        }
    },
    'West Bengal': {
        'Tier 1': {
            'Kolkata': {
                'localities': {
                    'Prime / Central': ['Park Street', 'Ballygunge', 'Alipore', 'Salt Lake Sector 5'],
                    'Standard Residential': ['New Town Action Area 1', 'Garia', 'Dum Dum Park', 'Behala'],
                    'Affordable Suburbs': ['Rajarhat', 'Barasat', 'Joka', 'Howrah Shibpur']
                },
                'sqft_rate': {'Prime / Central': (20, 32), 'Standard Residential': (13, 20), 'Affordable Suburbs': (8, 13)}
            }
        },
        'Tier 2': {
            'Siliguri': {
                'localities': {'Prime / Central': ['Sevoke Road', 'Hakim Para'], 'Standard Residential': ['Pradhan Nagar', 'Matigara'], 'Affordable Suburbs': ['Bagdogra', 'Deshbandhu Para']},
                'sqft_rate': {'Prime / Central': (12, 17), 'Standard Residential': (8, 13), 'Affordable Suburbs': (6, 9)}
            }
        }
    },
    'Gujarat': {
        'Tier 1': {
            'Ahmedabad': {
                'localities': {
                    'Prime / Central': ['Bodakdev', 'Satellite', 'Vastrapur', 'Prahlad Nagar'],
                    'Standard Residential': ['Thaltej', 'Navrangpura', 'Maninagar', 'Bopal'],
                    'Affordable Suburbs': ['Chandkheda', 'Gota', 'Naroda', 'Nikol']
                },
                'sqft_rate': {'Prime / Central': (18, 28), 'Standard Residential': (13, 19), 'Affordable Suburbs': (9, 13)}
            }
        },
        'Tier 2': {
            'Surat': {
                'localities': {'Prime / Central': ['Vesu', 'Piplod', 'Adajan'], 'Standard Residential': ['City Light', 'Pal', 'Rander'], 'Affordable Suburbs': ['Varachha', 'Katargam']},
                'sqft_rate': {'Prime / Central': (15, 23), 'Standard Residential': (11, 16), 'Affordable Suburbs': (8, 12)}
            },
            'Vadodara': {
                'localities': {'Prime / Central': ['Alkapuri', 'Vasna Road'], 'Standard Residential': ['Fatehgunj', 'Manjalpur', 'Gotri'], 'Affordable Suburbs': ['Waghodia Road', 'Gorwa']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (7, 10)}
            }
        }
    },
    'Rajasthan': {
        'Tier 2': {
            'Jaipur': {
                'localities': {
                    'Prime / Central': ['C-Scheme', 'Civil Lines', 'Malviya Nagar'],
                    'Standard Residential': ['Vaishali Nagar', 'Mansarovar', 'Tonk Road', 'Raja Park'],
                    'Affordable Suburbs': ['Jagatpura', 'Pratap Nagar', 'Ajmer Road', 'Sanganer']
                },
                'sqft_rate': {'Prime / Central': (16, 24), 'Standard Residential': (11, 17), 'Affordable Suburbs': (8, 12)}
            },
            'Udaipur': {
                'localities': {'Prime / Central': ['Fatehpura', 'Panchwati'], 'Standard Residential': ['Hiran Magri', 'Shobhagpura'], 'Affordable Suburbs': ['Bhuwana', 'Sukher']},
                'sqft_rate': {'Prime / Central': (14, 20), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 10)}
            }
        }
    },
    'Uttar Pradesh': {
        'Tier 2': {
            'Lucknow': {
                'localities': {
                    'Prime / Central': ['Gomti Nagar Extension', 'Hazratganj', 'Aliganj', 'Vibhuti Khand'],
                    'Standard Residential': ['Indira Nagar', 'Mahanagar', 'Ashiyana', 'Jankipuram'],
                    'Affordable Suburbs': ['Chinhat', 'Telibagh', 'Dubagga', 'Faizabad Road']
                },
                'sqft_rate': {'Prime / Central': (16, 25), 'Standard Residential': (11, 17), 'Affordable Suburbs': (7, 12)}
            },
            'Kanpur': {
                'localities': {'Prime / Central': ['Swaroop Nagar', 'Civil Lines'], 'Standard Residential': ['Kakadeo', 'Sharda Nagar'], 'Affordable Suburbs': ['Kalyanpur', 'Barra']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (6, 10)}
            },
            'Varanasi': {
                'localities': {'Prime / Central': ['Cantonment', 'Sigra'], 'Standard Residential': ['Lanka', 'Mahmoorganj'], 'Affordable Suburbs': ['Shivpur', 'Pandeypur']},
                'sqft_rate': {'Prime / Central': (13, 18), 'Standard Residential': (9, 14), 'Affordable Suburbs': (6, 10)}
            }
        }
    },
    'Madhya Pradesh': {
        'Tier 2': {
            'Indore': {
                'localities': {
                    'Prime / Central': ['Vijay Nagar', 'Saket Nagar', 'Palasia'],
                    'Standard Residential': ['AB Road', 'Bhawarkua', 'Rau', 'Bengali Square'],
                    'Affordable Suburbs': ['Super Corridor', 'Bypass Road', 'Nipania', 'Rajendra Nagar']
                },
                'sqft_rate': {'Prime / Central': (15, 23), 'Standard Residential': (10, 16), 'Affordable Suburbs': (7, 11)}
            },
            'Bhopal': {
                'localities': {'Prime / Central': ['Arera Colony', 'MP Nagar'], 'Standard Residential': ['Kolar Road', 'Shahpura'], 'Affordable Suburbs': ['Hoshangabad Road', 'Ayodhya Bypass']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (6, 10)}
            }
        }
    },
    'Kerala': {
        'Tier 2': {
            'Kochi': {
                'localities': {
                    'Prime / Central': ['Marine Drive', 'Panampilly Nagar', 'Kakkanad Infopark'],
                    'Standard Residential': ['Edappally', 'Kaloor', 'Palarivattom', 'Vyttila'],
                    'Affordable Suburbs': ['Aluva', 'Thrikkakara', 'Tripunithura', 'Kalamassery']
                },
                'sqft_rate': {'Prime / Central': (17, 26), 'Standard Residential': (12, 18), 'Affordable Suburbs': (8, 13)}
            },
            'Thiruvananthapuram': {
                'localities': {'Prime / Central': ['Kowdiar', 'Vazhuthacaud', 'Technopark Kazhakkoottam'], 'Standard Residential': ['Pattom', 'Sasthamangalam'], 'Affordable Suburbs': ['Sreekaryam', 'Mannanthala']},
                'sqft_rate': {'Prime / Central': (15, 23), 'Standard Residential': (11, 16), 'Affordable Suburbs': (7, 11)}
            }
        }
    },
    'Punjab': {
        'Tier 2': {
            'Chandigarh': {
                'localities': {
                    'Prime / Central': ['Sector 8', 'Sector 9', 'Sector 10', 'Sector 35'],
                    'Standard Residential': ['Sector 22', 'Sector 38', 'Sector 44', 'Mohali Phase 7'],
                    'Affordable Suburbs': ['Zirakpur VIP Road', 'Kharar', 'New Chandigarh', 'Dera Bassi']
                },
                'sqft_rate': {'Prime / Central': (22, 35), 'Standard Residential': (15, 23), 'Affordable Suburbs': (10, 15)}
            },
            'Ludhiana': {
                'localities': {'Prime / Central': ['Sarabha Nagar', 'Model Town'], 'Standard Residential': ['BRS Nagar', 'Civil Lines'], 'Affordable Suburbs': ['Ferozepur Road', 'Dugri']},
                'sqft_rate': {'Prime / Central': (14, 20), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 10)}
            }
        }
    },
    'Bihar': {
        'Tier 2': {
            'Patna': {
                'localities': {
                    'Prime / Central': ['Boring Road', 'Patliputra Colony', 'Kankarbagh'],
                    'Standard Residential': ['Bailey Road', 'Raja Bazar', 'Ashiana Nagar'],
                    'Affordable Suburbs': ['Danapur', 'Saguna More', 'Gola Road', 'Phulwari Sharif']
                },
                'sqft_rate': {'Prime / Central': (14, 21), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 11)}
            }
        }
    },
    'Odisha': {
        'Tier 2': {
            'Bhubaneswar': {
                'localities': {
                    'Prime / Central': ['Saheed Nagar', 'Jayadev Vihar', 'Patia Infocity'],
                    'Standard Residential': ['Nayapalli', 'Khandagiri', 'Chandrasekharpur'],
                    'Affordable Suburbs': ['Sundarpada', 'Kalinga Nagar', 'Pahal']
                },
                'sqft_rate': {'Prime / Central': (15, 22), 'Standard Residential': (10, 16), 'Affordable Suburbs': (7, 11)}
            }
        }
    },
    'Andhra Pradesh': {
        'Tier 2': {
            'Visakhapatnam': {
                'localities': {'Prime / Beach Front': ['MVP Colony', 'Siripuram', 'Pandurangapuram'], 'Standard Residential': ['Seethammadhara', 'Madhurawada IT SEZ', 'Gajuwaka'], 'Affordable Suburbs': ['Pendurthi', 'Sheela Nagar']},
                'sqft_rate': {'Prime / Beach Front': (16, 24), 'Standard Residential': (11, 16), 'Affordable Suburbs': (7, 11)}
            },
            'Vijayawada': {
                'localities': {'Prime / Central': ['Benz Circle', 'Moghalrajpuram'], 'Standard Residential': ['Governorpet', 'Poranki'], 'Affordable Suburbs': ['Gannavaram', 'Kanuru']},
                'sqft_rate': {'Prime / Central': (14, 20), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 10)}
            }
        }
    },
    'Uttarakhand': {
        'Tier 2': {
            'Dehradun': {
                'localities': {'Prime / Foothills': ['Rajpur Road', 'Dalanwala', 'Vasant Vihar'], 'Standard Residential': ['Sahastradhara Road', 'Chakrata Road', 'Subhash Nagar'], 'Affordable Suburbs': ['Prem Nagar', 'Kashimpur', 'Harrawala']},
                'sqft_rate': {'Prime / Foothills': (17, 26), 'Standard Residential': (11, 17), 'Affordable Suburbs': (8, 12)}
            }
        }
    },
    'Goa': {
        'Tier 2': {
            'Panaji': {
                'localities': {'Prime / Coastal': ['Miramar', 'Dona Paula', 'Caranzalem'], 'Standard Residential': ['Porvorim', 'Campal', 'Fontainhas'], 'Affordable Suburbs': ['Santa Cruz', 'Bambolim', 'Mapusa']},
                'sqft_rate': {'Prime / Coastal': (24, 38), 'Standard Residential': (16, 24), 'Affordable Suburbs': (11, 16)}
            }
        }
    },
    'Assam': {
        'Tier 2': {
            'Guwahati': {
                'localities': {'Prime / Central': ['GS Road', 'Christian Basti', 'Ulubari'], 'Standard Residential': ['Zoo Road', 'Beltola', 'Dispur'], 'Affordable Suburbs': ['Jalukbari', 'Six Mile', 'Lokhra']},
                'sqft_rate': {'Prime / Central': (14, 21), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 11)}
            }
        }
    },
    'Himachal Pradesh': {
        'Tier 3': {
            'Shimla': {
                'localities': {'Prime / Ridge': ['Mall Road', 'Chhota Shimla', 'Sanjauli'], 'Standard Residential': ['Kasumpti', 'New Shimla', 'Khalini'], 'Affordable Suburbs': ['Totu', 'Dhalli', 'Boileauganj']},
                'sqft_rate': {'Prime / Ridge': (18, 28), 'Standard Residential': (12, 18), 'Affordable Suburbs': (8, 12)}
            }
        }
    },
    'Jharkhand': {
        'Tier 2': {
            'Ranchi': {
                'localities': {'Prime / Central': ['Morabadi', 'Kanke Road', 'Ashok Nagar'], 'Standard Residential': ['Doranda', 'Lalpur', 'Bariatu'], 'Affordable Suburbs': ['Tatisilwai', 'Namkum', 'Ratu Road']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (6, 10)}
            }
        }
    },
    'Chhattisgarh': {
        'Tier 2': {
            'Raipur': {
                'localities': {'Prime / Central': ['Shankar Nagar', 'Civil Lines', 'Telibandha'], 'Standard Residential': ['Devendra Nagar', 'Samta Colony', 'Tatibandh'], 'Affordable Suburbs': ['Kachna', 'Bhanpuri', 'Dharampura']},
                'sqft_rate': {'Prime / Central': (13, 19), 'Standard Residential': (9, 14), 'Affordable Suburbs': (6, 10)}
            }
        }
    },
    'Jammu & Kashmir': {
        'Tier 3': {
            'Srinagar': {
                'localities': {'Prime / Boulevard': ['Rajbagh', 'Hyderpora', 'Karan Nagar'], 'Standard Residential': ['Sanat Nagar', 'Jawahar Nagar', 'Bemina'], 'Affordable Suburbs': ['Soura', 'Rawalpora', 'Nowgam']},
                'sqft_rate': {'Prime / Boulevard': (15, 23), 'Standard Residential': (10, 15), 'Affordable Suburbs': (7, 11)}
            }
        }
    }
}

def generate_pan_india_dataset():
    records = []
    furnish_options = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    furnish_weights = [0.25, 0.45, 0.30]
    tenant_options = ['Bachelors/Family', 'Bachelors', 'Family']
    tenant_weights = [0.60, 0.20, 0.20]

    bhk_specs = [
        # BHK, Bathrooms range, Size range (sqft)
        (1, [1], (350, 650)),
        (2, [1, 2], (650, 1150)),
        (3, [2, 3], (1100, 1750)),
        (4, [3, 4], (1650, 2600))
    ]

    for state, tiers in STATE_CITY_CONFIG.items():
        for tier_name, cities in tiers.items():
            for city_name, city_info in cities.items():
                localities_dict = city_info['localities']
                sqft_rates = city_info['sqft_rate']

                # Generate 120-180 realistic listings per city
                n_listings = random.randint(120, 180)
                for _ in range(n_listings):
                    # Pick locality zone
                    zone = random.choice(list(localities_dict.keys()))
                    locality = random.choice(localities_dict[zone])
                    rate_min, rate_max = sqft_rates[zone]

                    # Pick BHK config
                    bhk, possible_baths, (size_min, size_max) = random.choice(bhk_specs)
                    baths = random.choice(possible_baths)
                    size = random.randint(size_min, size_max)

                    # Calculate base rent per sqft with slight variation
                    sqft_rate = random.uniform(rate_min, rate_max)

                    # Adjustment factors
                    furnish = random.choices(furnish_options, weights=furnish_weights)[0]
                    furnish_mult = 1.25 if furnish == 'Furnished' else (1.08 if furnish == 'Semi-Furnished' else 1.0)

                    tenant = random.choices(tenant_options, weights=tenant_weights)[0]
                    bath_mult = 1.0 + (0.05 * (baths - 1))

                    raw_rent = size * sqft_rate * furnish_mult * bath_mult

                    # Round to nearest 500
                    final_rent = int(round(raw_rent / 500.0) * 500)
                    # Filter out anything > 100,000 to keep within our cap
                    if final_rent > 100000:
                        final_rent = 95000

                    records.append({
                        'State': state,
                        'City': city_name,
                        'City_Tier': tier_name,
                        'Locality_Zone': zone,
                        'Area Locality': locality,
                        'BHK': bhk,
                        'Bathroom': baths,
                        'Size': size,
                        'Furnishing Status': furnish,
                        'Tenant Preferred': tenant,
                        'Rent': final_rent
                    })

    df_expanded = pd.DataFrame(records)
    print(f"Generated {len(df_expanded)} realistic listings across {df_expanded['State'].nunique()} States and {df_expanded['City'].nunique()} Cities!")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'data', 'raw', 'cleaned_data.csv')
    df_expanded.to_csv(output_path, index=False)
    print(f"Successfully saved to {output_path}")

if __name__ == '__main__':
    generate_pan_india_dataset()
