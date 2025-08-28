import requests
import json
payload = {
    "request_id": '8766',
    "input_polygon": [
  [
    33.041376,
    34.653923
  ],
  [
    33.031083,
    34.636411
  ],
  [
    33.050982,
    34.625676
  ],
  [
    33.048923,
    34.647709
  ]
],
    #     [
    #   [3672531.636, 4113070.473],
    #   [3683060.901, 4120108.645],
    #   [3684952.665, 4111452.193],
    #   [3671518.851, 4111222.930],
    #   [3672531.636, 4120070.473]
    # ],
    
    "use_case": "aquaculture",
    "datasets": {
        "aquaculture": {
            "geom": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [3742983.8243410653, 4141365.1805400824],
                                    [3739287.902042941, 4137499.0113585845],
                                    [3742922.612075378, 4134993.865558433],
                                    [3748359.448303795, 4140785.3609113554],
                                    [3746214.9561182486, 4144193.6841967963],
                                    [3742983.8243410653, 4141365.1805400824]
                                ]
                            ]
                        }
                    }
                ]
            }
        }
    },
    "models": ["aug","pf","ws"],
    "model": {"model_AUG": "","model_WS": "", "model_PF": ""},
    "C_number": 3,
    "N_size": [3, 10],
    "calc_type": "dist",
    "thresholds": {
        "Coastline": {"thresholds": [2500, 4000, 5000, 8000, 10000], "weights": 0.3,"enabled": True},
        "WindSpeed": {"thresholds": [2, 3, 4, 4.5, 5], "weights": 0.3, "enabled": True},
        "Bathymetry": {"thresholds": [-100, -80, -60, -50, -30], "weights": 0.1, "enabled": True},
        "Ports": {"thresholds": [2500, 4000, 5000, 8000, 10000], "weights": 0.1, "enabled": True},
        "aquaculture": {"thresholds": [12, 32, 43, 62, 534], "weights": 0.2, "enabled": True},
    },
    "threads": {
      "threads": 12,
      "omp_num_threads": 12,
      "mkl_num_threads": 12,
      "openblas_threads": 12
    }
}
new_payload = {
    "request_id": "4875",
    "input_polygon": [
  [
    33.514793,
    34.771985
  ],
  [
    32.603484,
    34.650062
  ],
  [
    32.694066,
    34.274138
  ],
  [
    33.981427,
    34.376213
  ]
    ],
    "use_case": "aquaculture",
    "model": {
        "model_AUG": True,
        "model_WS": False,
        "model_PF": True
    },
    "C_number": 3,
    "N_size": [
        3,
        10
    ],
    "datasets": {},
    "thresholds": {
        "Coastline": {
            "thresholds": [
                2500,
                4000,
                5000,
                8000,
                10000
            ],
            "weights": 0.2,
            "enabled": True
        },
        "WindSpeed": {
            "thresholds": [
                2,
                3,
                4,
                4.5,
                5
            ],
            "weights": 0.3,
            "enabled": True
        },
        "Bathymetry": {
            "thresholds": [
                -100,
                -80,
                -60,
                -50,
                -30
            ],
            "weights": 0.1,
            "enabled": True
        },
        "Ports": {
            "thresholds": [
                2500,
                4000,
                5000,
                8000,
                10000
            ],
            "weights": 0.1,
            "enabled": True
        },
        "Habitat_map_clean": {
            "thresholds": [
                10,
                20,
                30,
                40,
                50
            ],
            "weights": 0.1,
            "source": "server",
            "enabled": True,
            "fileId": "6cd5dfc1-b1b4-48fb-8ea2-9f8356aa5e53"
        },
        "Ports_custom": {
            "thresholds": [
                10,
                20,
                30,
                40,
                50
            ],
            "weights": 0.1,
            "source": "server",
            "fileId": "bbc1533d-87d8-4e0b-9c78-35e9ffec2809"
        },
        "Bathymetry_custom": {
            "thresholds": [
                10,
                20,
                30,
                40,
                50
            ],
            "weights": 0.1,
            "source": "server",
            "fileId": "ee67a522-900e-48f7-9aa1-f8f99fa213b6"
        }
    },
    "calc_type": "dist",
}
very_largePolygon =[
                    [
                        33.514793,
                        34.771985
                    ],
                    [
                        32.603484,
                        34.650062
                    ],
                    [
                        32.694066,
                        34.274138
                    ],
                    [
                        33.981427,
                        34.376213
                    ]
                        ]
resp = requests.post("http://localhost:8000/api/v1/run", json=payload)
print(resp.status_code, resp.json())
# save result to file
with open("./result.json", "w") as f:
    json.dump(resp.json(), f, indent=2)
