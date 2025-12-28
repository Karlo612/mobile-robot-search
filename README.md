# mobile-robot-search
implementation of Astar, BFS and DFS in grid base environment for mobile robot


#environment setup

1. go to the project directory
2. create virtual environment:
   python -m venv venv
3. activate virtual entironment:
    source venv/bin/activate
4. install required library:
    pip install -r requirements.txt
5. configure config.json based on what you want to run see detals below:
    a) you can run on Navigation 'mode' = 'nav'. this means it will read from excel and run only the selected planner in the config file . 

    b) you can run on benchmark "mode": = bench . in this case there will be no simulation. the code will gererate randon grid . small , medium and large . the perform test autometically for each type of search and draw a performance graph. 

6. run code from the project directory using below command:
    python -m CodeBase.main

#config.json

{
  "map_file": "Data/maps/map16x22.xlsx",
  "resolution": 1.0,
  "origin": [0, 0],
  "robot_radius": 2, <change this to set robot radious>

  "start_grid": [2, 3], <change this to set robot position when running in nav mode>
  "goal_grid": [13, 18], <change this to set robot position when running in nav mode>
  
  "motion": "8n", <change this to set robot motion >
  "planner": "BFS", <change this to select which type of search you want to run>
  "visualize_search": true, <this is for nav mode only>
  "use_tree_search": true, <if you make this 'true' it will run in tree based mode . otherwise graphbased>
  "mode": "nav" <change this 'nav' mean single mode and 'bench' means conparisn mode>
}



