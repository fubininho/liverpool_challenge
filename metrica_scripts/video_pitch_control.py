import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from metrica_scripts import Metrica_IO as mio
from metrica_scripts import Metrica_Viz as mviz
from metrica_scripts import Metrica_Velocities as mvel

from moviepy import editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

class player(object):

   # player object holds position, velocity
   def __init__(self,player_id,team,teamname,params={'amax':7,'vmax':5,'ttrl_sigma':0.54},xgrid=50,ygrid=32):
      self.id = player_id
      self.teamname = teamname
      self.playername = "%s_%s_" % (teamname,player_id)
      self.get_position(team)
      self.get_velocity(team)
      self.vmax = params['vmax']
      self.amax = params['amax']
      self.reaction_time = params['vmax']/params['amax']
      self.ttrl_sigma=params['ttrl_sigma'] # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
      self.pitch_control = 0. # initialise this for later
      self.pitch_control_surface = np.zeros( shape = (ygrid, xgrid) )
      
   def get_position(self,team):
      self.position = np.array( [ team[self.playername+'x'], team[self.playername+'y'] ] )
      self.inframe = not np.any( np.isnan(self.position) )
   
   def get_velocity(self,team):
      self.velocity = np.array( [ team[self.playername+'vx'], team[self.playername+'vy'] ] )
      if np.any( np.isnan(self.velocity) ):
         self.velocity = np.array([0.,0.])

   def simple_time_to_reach_location(self,location):
      reaction_time=self.vmax/self.amax
      r_reaction = self.position + self.velocity*reaction_time
      arrival_time = reaction_time + np.linalg.norm(location-r_reaction)/self.vmax
      self.time_to_reach_location=arrival_time
      return(arrival_time)

   def improved_time_to_reach_location(self,location):
      Xf=location
      X0=self.position
      V0=self.velocity
      alpha = self.amax/self.vmax
      
      #equations of motion + equation 3 from assumption that the player accelerate 
      #with constant acceleration amax to vmax
      #we have to add abs(t) to make t be positive
      def equations(p):
         vxmax, vymax, t = p
         eq1 = Xf[0] - (X0[0] + vxmax*(abs(t) - (1 - np.exp(-alpha*abs(t)))/alpha)+((1 - np.exp(-alpha*abs(t)))/alpha)*V0[0])
         eq2 = Xf[1] - (X0[1] + vymax*(abs(t) - (1 - np.exp(-alpha*abs(t)))/alpha)+((1 - np.exp(-alpha*abs(t)))/alpha)*V0[1])
         eq3 = np.sqrt(vxmax**2+vymax**2) - self.vmax
         return (eq1,eq2,eq3)
      
      #prediction for three unknowns
      t_predict=np.linalg.norm(Xf-X0)/self.vmax+0.7
      v_predict=self.vmax*(Xf-X0)/np.linalg.norm(Xf-X0)
      vxmax, vymax, t =  fsolve(equations, (v_predict[0], v_predict[1], t_predict))

      self.time_to_reach_location=abs(t)
      
      return(abs(t))
   
   def probability_to_reach_location(self,T):
      f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.ttrl_sigma * (T-self.time_to_reach_location ) ) )
      return f


def parameters():
   """
   default_model_params()
   
   Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
   
   Parameters
   -----------
   time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
   
   
   Returns
   -----------
   
   params: dictionary of parameters required to determine and calculate the model
   
   """
   # key parameters for the model, as described in Spearman 2018
   params = {}
   # model parameters
   params['amax'] = 7. # maximum player acceleration m/s/s
   params['vmax'] = 5. # maximum player speed m/s
   params['ttrl_sigma'] = 0.54 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
   params['kappa_def'] =  1. # kappa parameter in Spearman 2018 that gives the advantage defending players to control ball
   params['lambda_att'] = 3.99 # ball control parameter for attacking team
   params['lambda_def'] = 3.99 * params['kappa_def'] # ball control parameter for defending team
   params['average_ball_speed'] = 15. # average ball travel speed in m/s
   # numerical parameters for model evaluation
   params['int_dt'] = 0.04 # integration timestep (dt)
   params['max_int_time'] = 10 # upper limit on integral time
   params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
   # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
   # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
   params['time_to_control_att'] = 3*np.log(10) * (np.sqrt(3)*params['ttrl_sigma']/np.pi + 1/params['lambda_att'])
   params['time_to_control_def'] = 3*np.log(10) * (np.sqrt(3)*params['ttrl_sigma']/np.pi + 1/params['lambda_def'])
   
   # sigma normal distribution for relevant pitch control
   params['sigma_normal'] = 23.9
   # alpha : dependence of the decision conditional probability by the PPCF
   params['alpha'] = 1.04
   
   return params


def attacking_team_frame(events,frame):
   #the game doesn't start at frame 0, so we check that the frame asked is superior to the frame when game starts
   
   attacking_team=events[((events.Type=="RECOVERY") | (events.Type=="SET PIECE")) & (events['Start Frame']<=frame)]['Team'].values[-1] 

   return(attacking_team)


def where_home_team_attacks(home_team,events):
   '''
   Determines where teams attack on the first period using team x average position at game start
   
   Returns
   -------
      -1 if home team attacks on the left (x<0)
      1 if home team attacks on the right (x>0)
   
   '''
   game_start_frame=events.iloc[0]['Start Frame']
   home_team_x_cols=[c for c in home_team.columns if c.split('_')[-1]=='x' and c.split('_')[-2]!='ball']

   if home_team.loc[game_start_frame,home_team_x_cols].mean()>0:
      return(-1)
   else:
      return(1)


def find_offside_players(attacking_players, defending_players, where_attack, ball_pos):
   '''
   Determines which attacking players are in offside position. 
   A player is caught offside if he’s nearer to the opponent’s goal 
   than both the ball and the second-last opponent (including the goalkeeper).
   
   Returns
   -------
      offside_players : the list of offside players names
   '''
   
   offside_players=[]
   # if attacking team attacks on the right
   if where_attack==1:
      
      #find the second-last defender
      x_defending_players=[]
      for player in defending_players:
         x_defending_players.append(player.position[0])
      x_defending_players=np.sort(x_defending_players)
      second_last_defender_x=x_defending_players[-2]
      
      for player in attacking_players:
         position=player.position
         #if player is nearer to the opponent's goal than the ball
         if position[0]>ball_pos[0] and position[0]>second_last_defender_x:
               offside_players.append(player)
   
   # if attacking team attacks on the right
   if where_attack==-1:
      
      #find the second-last defender
      x_defending_players=[]
      for player in defending_players:
         x_defending_players.append(player.position[0])
      x_defending_players=np.sort(x_defending_players)
      second_last_defender_x=x_defending_players[1]
      
      for player in attacking_players:
         position=player.position
         #if player is nearer to the opponent's goal than the ball
         if position[0]<ball_pos[0] and position[0]<second_last_defender_x:
               offside_players.append(player.playername)
               
   return(offside_players)


def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, where_attack, params):
   """ calculate_pitch_control_at_target
   
   Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
   
   Parameters
   -----------
      target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
      attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
      defending_players: list of 'player' objects (see player class above) for the players on the defending team
      ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
      where_attack: where attacking team attacks (1 on the right, -1 on the left)
      params: Dictionary of model parameters
      
   Returns
   -----------
      PPCFatt: Pitch control probability for the attacking team
      PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
   """
   # calculate ball travel time from start position to end position.
   if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
      ball_travel_time = 0.0 
   else:
      # ball travel time is distance to target position from current ball position divided assumed average ball speed
      ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
      
   # find offside attacking players
   offside_players=find_offside_players(attacking_players, defending_players, where_attack, ball_start_pos)
   
   # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity) (if player isn't offside)
   tau_min_att = np.nanmin( [p.improved_time_to_reach_location(target_position) for p in attacking_players if p.playername not in offside_players] )
   tau_min_def = np.nanmin( [p.improved_time_to_reach_location(target_position) for p in defending_players] )
   
   # check whether we actually need to solve equation 
   if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
      # if defending team can arrive significantly before attacking team, no need to solve pitch control model
      return 0., 1.
   elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
      # if attacking team can arrive significantly before defending team, no need to solve pitch control model
      return 1., 0.
   else: 
      # solve pitch control model by integrating equation 3 in Spearman et al.
      # first remove any player that is far (in time) from the target location
      # remove offside players
      attacking_players = [p for p in attacking_players if (p.playername not in offside_players) and (p.time_to_reach_location-tau_min_att < params['time_to_control_att']) ]
      defending_players = [p for p in defending_players if p.time_to_reach_location-tau_min_def < params['time_to_control_def'] ]
      
      # set up integration arrays
      dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
      PPCFatt = np.zeros_like( dT_array )
      PPCFdef = np.zeros_like( dT_array )
      
      # set PPCF to 0. for each player
      for player in attacking_players:
         player.pitch_control=0.
      for player in defending_players:
         player.pitch_control=0.
      
      # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
      ptot = 0.0
      i = 1
      while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
         T = dT_array[i]
         for player in attacking_players:
               
            # calculate lambda for 'player' (0 if offside)
            if player.playername in offside_players:
               lambda_att=0
            else:
               lambda_att=params['lambda_att']
               
            # calculate ball control probablity for 'player' in time interval T+dt
            dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_to_reach_location( T ) * lambda_att
            
            # make sure it's greater than zero
            assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'

            player.pitch_control += dPPCFdT*params['int_dt'] # total contribution from individual player
            PPCFatt[i] += player.pitch_control # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            
         for player in defending_players:
            # calculate ball control probablity for 'player' in time interval T+dt
            dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_to_reach_location( T ) * params['lambda_def']
            
            # make sure it's greater than zero
            assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'

            player.pitch_control += dPPCFdT*params['int_dt'] # total contribution from individual player
            PPCFdef[i] += player.pitch_control # add to sum over players in the defending team
            
         ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
         i += 1
      if i>=dT_array.size:
         print("Integration failed to converge: %1.3f" % (ptot) )
         
      return PPCFatt[i-1], PPCFdef[i-1]


def generate_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params, field_dimen = (106.,68.,), n_grid_cells_x = 50, return_players=False):
   """ generate_pitch_control_for_frame
   
   Evaluates pitch control surface over the entire field at the moment of the given frame
   
   Parameters
   -----------
      frame: instant at which the pitch control surface should be calculated
      tracking_home: tracking DataFrame for the Home team
      tracking_away: tracking DataFrame for the Away team
      events: Dataframe containing the event data
      params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                     n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
      
   Returrns
   -----------
      PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
            Surface for the defending team is just 1-PPCFa.
      xgrid: Positions of the pixels in the x-direction (field length)
      ygrid: Positions of the pixels in the y-direction (field width)
   """
   # get the details of the event (team in possession, ball_start_position, where team in possession attacks)
   attacking_team = attacking_team_frame(events,frame)
   assert attacking_team=='Home' or attacking_team=='Away', 'attacking team should be Away or Home'
   
   ball_start_pos = np.array( [ tracking_home.loc[frame]['ball_x'], tracking_home.loc[frame]['ball_y'] ] )
   
   where_home_attacks = where_home_team_attacks(tracking_home,events)
   period=tracking_home.loc[frame]['Period']
   if attacking_team=='Home':
      if period==1:
         where_attack=where_home_attacks
      else:
         where_attack=-where_home_attacks
   else:
      if period==1:
         where_attack=-where_home_attacks
      else:
         where_attack=where_home_attacks
      
   # break the pitch down into a grid
   n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
   xgrid = np.linspace( -field_dimen[0]/2., field_dimen[0]/2., n_grid_cells_x)
   ygrid = np.linspace( -field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y )
   
   # initialise pitch control grids for attacking and defending teams 
   PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
   PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
   # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
   if attacking_team=='Home':
      attacking_players = initialise_players(tracking_home.loc[frame],'Home',params, xgrid=len(xgrid), ygrid=len(ygrid))
      defending_players = initialise_players(tracking_away.loc[frame],'Away',params, xgrid=len(xgrid), ygrid=len(ygrid))
   else:
      defending_players = initialise_players(tracking_home.loc[frame],'Home',params, xgrid=len(xgrid), ygrid=len(ygrid))
      attacking_players = initialise_players(tracking_away.loc[frame],'Away',params, xgrid=len(xgrid), ygrid=len(ygrid))

   # calculate pitch control model at each location on the pitch
   # if we want to save individual pitch control
   if return_players:
      for i in range( len(ygrid) ):
         for j in range( len(xgrid) ):
               target_position = np.array( [xgrid[j], ygrid[i]] )
               PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, where_attack, params)
               for player in attacking_players:
                  player.pitch_control_surface[i,j] = player.pitch_control
               for player in defending_players:
                  player.pitch_control_surface[i,j] = player.pitch_control
   else:
      for i in range( len(ygrid) ):
         for j in range( len(xgrid) ):
               target_position = np.array( [xgrid[j], ygrid[i]] )
               PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, where_attack, params)

   # check probabilitiy sums within convergence
   checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
   assert abs(1-checksum) < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)

   if return_players:
      players_pitch_control = {}
      for player in attacking_players:
         players_pitch_control[player.playername] = player.pitch_control_surface
      for player in defending_players:
         players_pitch_control[player.playername] = player.pitch_control_surface
         
      return(PPCFa,xgrid,ygrid,players_pitch_control)
   
   return PPCFa,xgrid,ygrid


def plot_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params, figax=None, alpha = 0.7, include_player_velocities=True, annotate=True, field_dimen = (106.,68.,), n_grid_cells_x = 50):
   """ plot_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params ,PPCF, xgrid, ygrid )
   
   Plots the pitch control surface at the instant of the frame. Player and ball positions are overlaid.
   
   Parameters
   -----------
      frame: the instant at which the pitch control surface should be calculated
      events: Dataframe containing the event data
      tracking_home: (entire) tracking DataFrame for the Home team
      tracking_away: (entire) tracking DataFrame for the Away team
      PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
      xgrid: Positions of the pixels in the x-direction (field length) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
      ygrid: Positions of the pixels in the y-direction (field width) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
      alpha: alpha (transparency) of player markers. Default is 0.7
      include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
      annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      
   Returns
   -----------
      fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
   """  
   # plot frame and event
   if figax is None: # create new pitch 
      fig,ax = mviz.plot_pitch(field_dimen=field_dimen, field_color= 'white')
   else: # overlay on a previously generated pitch
      fig,ax = figax # unpack tuple
   mviz.plot_frame( tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
   
   #generate pitch control
   PPCF, xgrid, ygrid=generate_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params, field_dimen = field_dimen, n_grid_cells_x = n_grid_cells_x)
   
   #find attacking team
   attacking_team = attacking_team_frame(events,frame)
   
   # plot pitch control surface
   if attacking_team=='Home':
      cmap = 'bwr'
   else:
      cmap = 'bwr_r'
   ax.imshow(np.flipud(PPCF), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),interpolation='hanning',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
   
   return fig,ax


def initialise_players(team,teamname,params={'amax':7,'vmax':5},xgrid=50,ygrid=32):    
   # get player  ids
   player_ids = np.unique( [ c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
   # create list
   team_players = []
   for p in player_ids:
      # create a player object for player_id 'p'
      team_player = player(p,team,teamname,params=params)
      if team_player.inframe:
         team_players.append(team_player)
         
   return team_players


def distance_to_location(location,player):
   origin=player.position
   return(np.linalg.norm(location-origin))


def calculate_basic_pitch_control_at_location(location, home_players, away_players, distance_function = distance_to_location):
    
   #initialise distance to location
   distance=np.inf
   closest_player=None
   closest_team=None
   
   for player in home_players:
      #calculate distance from player to location
      d=distance_function(location,player)
      if d<distance:
         closest_player=player.playername
         closest_team=player.teamname
         distance=d
         
   for player in away_players:
      #calculate distance from player to location
      d=distance_function(location,player)
      if d<distance:
         closest_player=player.playername
         closest_team=player.teamname
         distance=d
   
   if closest_team=='Home':
      #if Home team is closest --> probability to possess the ball=1 while for away time proba=0
      return(1,0)
   
   if closest_team=='Away':
      return(0,1)


def draw_frame_x(hometeam,awayteam, t, players_to_annotate, events, params, figax=None, fps=25, 
                 team_colors=('r','b'), field_dimen = (106.0,68.0), 
                 include_player_velocities=True, PlayerMarkerSize=10, PlayerAlpha=0.7,dpi=100):
    
   #we have to convert t*fps to int because t is a float
   fig,ax=mviz.plot_pitch(field_color='white')
   fig,ax=mviz.plot_frame(hometeam.iloc[int(t*fps)], awayteam.iloc[int(t*fps)], figax=(fig,ax), 
                        team_colors=team_colors, field_dimen = field_dimen, 
                        include_player_velocities=include_player_velocities, PlayerMarkerSize=PlayerMarkerSize, PlayerAlpha=PlayerAlpha)
   
   fig,ax=plot_pitch_control_for_frame(int(t*fps),hometeam,awayteam,events,params,(fig,ax))
   
   # fig,ax=plot_basic_pitch_control_for_frame(hometeam,awayteam,int(t*fps),figax=(fig,ax))
   
   players_id=players_to_annotate.iloc[:,0].values
   nums=players_to_annotate.iloc[:,1].values
   for k in range(len(nums)):
      ax.text(hometeam.iloc[int(t*fps)]['Home_'+str(players_id[k])+'_x'],hometeam.iloc[int(t*fps)]['Home_'+str(players_id[k])+'_y'],str(int(nums[k])),fontsize=10,color='k',horizontalalignment='center', verticalalignment='center')
   
   image = mplfig_to_npimage(fig)
   plt.close()
   return image

def make_animation(hometeam,awayteam, fpath, players_to_annotate, events, fname='clip_test', figax=None, fps=25, 
                   team_colors=('r','b'), field_dimen = (106.0,68.0), 
                   include_player_velocities=True, PlayerMarkerSize=15, PlayerAlpha=0.7,dpi=100):
    
   length=len(hometeam)/fps
   params = parameters()

   clip = mpy.VideoClip(
      lambda x: draw_frame_x(hometeam, awayteam, x, players_to_annotate, events, params, figax=figax, team_colors=team_colors, field_dimen = field_dimen, include_player_velocities=include_player_velocities, PlayerMarkerSize=PlayerMarkerSize, PlayerAlpha=PlayerAlpha),
      duration=length).set_fps(fps)

   clip.write_videofile(fpath+fname+'.mp4')

def plot_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, fname=None, alpha = 0.7, include_player_velocities=True, annotate=True, field_dimen = (106.,68.,), n_grid_cells_x = 50):
   """ generate_pitch_control_for_event
   
   Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)
   
   Parameters
   -----------
      event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
      events: Dataframe containing the event data
      tracking_home: tracking DataFrame for the Home team
      tracking_away: tracking DataFrame for the Away team
      params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                     n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
      
   Returns
   -----------
      PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
            Surface for the defending team is just 1-PPCFa.
      xgrid: Positions of the pixels in the x-direction (field length)
      ygrid: Positions of the pixels in the y-direction (field width)
   """
   # pick a pass at which to generate the pitch control surface
   pass_frame = events.loc[event_id]['Start Frame']
   pass_team = events.loc[event_id].Team
   
   # plot frame and event
   fig,ax = mviz.plot_pitch(field_color='white', field_dimen = field_dimen)
   mviz.plot_frame( tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
   mviz.plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
   
   # generate pitch control
   PPCF, xgrid, ygrid=generate_pitch_control_for_frame(pass_frame, tracking_home, tracking_away, events, params, field_dimen = field_dimen, n_grid_cells_x = n_grid_cells_x)
   
   # plot pitch control surface
   if pass_team=='Home':
      cmap = 'bwr'
   else:
      cmap = 'bwr_r'
   ax.imshow(np.flipud(PPCF), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),interpolation='hanning',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
   
   if fname:
      # Save the full figure...
      plt.savefig("{}.png".format(fname))
   else:
      plt.show()

   plt.close()
   # return fig,ax