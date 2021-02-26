# FLAME GPU Example: Pedestrian Flood Navigation

A model which combines the flood model (see separate flood model example) with the FLAMGPU pedestrian navigation example.

CURRENT MODEL (update 29OCT2018)
in brief:
1- The model can simulate evacuation procedure (which is optional):
	1-1 by enabling the pedestrian hearing an emergency alarm (for early evacuation planning)
	1-2 or observing the upcoming water flow towards the area
2- The model can simualte different types of floods (based on Environment Agency's guidance document)
3- The state of pedestrians can be changed with respect to the characteristics of water flow, e.g. if a pedestrian get stuck in water or is considered as casualties
	3-1 dead or got stuck pedestrians cannot move anymore, assuming stuck ones are waiting for help
4- The pedestrians can put sandbags in front of water to block the water flow,
	4-1 the sizes of sandbags and proposed dike can be defined in input file
	4-2 the timing of sandbagging procedure is modifiable to optional values
	4-3 the area of pick up point to grab sandbags can be change to any location which corresponds to any exit
	4-4 drop point is only applicable to exit7 in the model (for the location of exit 7 see \iterations\init_ped_flood.jpg)
5- * The body of pedestrians can be considered as moving obstacles. This is optional for the user to choose.
6- Time-step of flooding can be either choosed adaptive or static
7- The speed of the simulation of pedestrian movement can be modified (e.g. can be increased where they are putting sandbags to accelerate the simulation time)
8- The location of inflow boundary in modifiable to any plece at boundaries

PLEASE NOTE **** Capabilities of current model will be explained in details in a separate file along with comprehensive instructions ****

Videos of some simulations can be found here: https://www.youtube.com/channel/UCAeXqodwipy2nXl30RFWk5Q 




An example input file which follows the above instructions is provided in iterations\map.xml

CURRENT EXAMPLE: flooding event in a hypothetical shopping centre

To change the initial data of water flow (e.g. severity) and pedestrian (e.g. population size), simply open interations/map.xml and change constant variables. 

To change initial condition of pedestrian model and flood please follow the instruction (generating other examples and change topography features):  
# Generating initial state files
TODO: 
1- Generate initial condition for pedestrian model on windows, To do: 
1-1 Open 'PlanEditor.sln' file located in 'FGPUGridNavPlanEditor' within MS Visual Studio (2015 +)
1-2 Set solution configuration on 'Debug' and 'Any CPU'
1-3 Press 'Start' to pop up the 'Floor Plan Editor' window
1-4 Press Define Area button to define a rectanglar domain representing an imaginary ground by clicking and streching the rectangle.
1-5 Press Draw Walls and click where the wall starts then continue until the end is again reached. Finalise drawing by double click 
    on the starting point. 
1-6 Press Define Exits to indicate where the exits are located. Exits are the location where pedestrians enter and/or leave the domain.
    to do so, draw lines within the area surrounded by walls which have been previously drew in 1-5
	NOTE: Exits must be located within a distance from walls since pedestrians cannot cross the walls. 
1-7 Define build map Width and Height, set it to number of agents in each direction e.g. 64 used in the current example. 
1-8 Press Build It! to export map.xml file which defines the initial state of navigation map (floor plan)
NOTE: Pedestrians use this data to assing initial state of variables in their memory. 
1-9 Open map.xml file and assign values to the constant variables. Use the data provided in the example as initial try.
1-10 After bulding the initial data for navmap agents, go to /iterations and copy the environment variables and agent data
     within init_ped.xml file to map.xml file

2- Generate initial condition for the flood model, To do, 
2-1 To generate initial conditon of the flooding open FloodPedestrian > Flood_XML_inpGen and open XML_inpGen project in Visal Studio (for Windows)
NOTE: this project contains only a source code which generates topography features within a domain of flood agents
      and similarly outputs the data as an xml file very similar to that of pedestrian model
2-2 Most of the global variables (in environment element) are defined within xmlGen.cpp source code, 
    a summarised descroption of each can be found as comments.
2-3 To change the topography, simply modify 'bed_data' function
    NOTE: The current example is set to create an imaginary shopping centre with all buildings and opennings
2-4 To change the flow characteristics only change the peak discharge which will be considered within a hydrograph

PLEASE CONSIDER: the initial data of water flow at inflow boundaries are only modifiable by changing the variables related to discharge
		and flood timing. Meaning that unlike the academic test-cases (dam-breaks) the inflow and boundary location
		can be changed to produce different types of floods

2-5 After setting all the constant variables, Build and rund the model. 
2-6 Then go to FloodPedestrian\Flood_XML_inpGen\XML_inpGen and open flood_init.txt 
2-7 Copy and paste all the constant variables and agent in iterations/map.xml file (in case of changing the current model, simply replace the prvious data with new agent data)
2-8 Copy/replace the environment variables. 


The model can be run in visualisation mode. By default pedestrians will be shown with navigation agents.
To view the water flow use the 'w' key and to see topograpy features use 'b' key.



