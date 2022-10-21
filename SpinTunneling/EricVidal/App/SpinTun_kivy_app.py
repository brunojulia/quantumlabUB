<exp_P>:
    Label:
        text: "You pressed the button"
        size_hint: 0.6, 0.2
        pos_hint: {"x":0.2, "top":1}

    Button:
        text: "You pressed the button"
        size_hint: 0.8, 0.2
        pos_hint: {"x":0.1, "y":0.1}

<Starting_Screen>:
    name: 'Starting'
    
    #Starting LAYOUT
    FloatLayout: 
		id: menu_layout
		
		#size: root.width, root.height #layout that occupies all image
		size_hint_x:1 
		size_hint_y:1
    	
    	#Sets bg color
    	canvas.before: 
			Color: 
				rgba: (1,1,1,1) 
			Rectangle: 
				size: self.size 
				pos: self.pos
				
        #App Logo
        GridLayout:
			cols: 1
			rows: 1 
			size_hint_x:1
			size_hint_y:1
			pos_hint:{"x":0,"y":0}
			Image:  
				source: "graphs/ST_logo.png"
				x:self.parent.x
				y:self.parent.y
				size: self.parent.width, self.parent.height 
				allow_stretch: True
		    	keep_ratio: True 
    
<Menu_Screen>:
    name: 'Menu'
    
    #MENU LAYOUT
    FloatLayout: 
		id: menu_layout
		
		#size: root.width, root.height #layout that occupies all image
		size_hint_x:1 
		size_hint_y:1
    	
    	#Sets bg color
    	canvas.before: 
			Color: 
				rgba: (1,1,1,1) 
			Rectangle: 
				size: self.size 
				pos: self.pos
				
        #TOP SCREEN
        FloatLayout: 
			size_hint_x: 1 
			size_hint_y:0.25
			pos_hint:{"x":0,"y":0.75}
			
			#App Logo
            GridLayout:
    			cols: 1
    			rows: 1 
    			size_hint_x:0.25
    			size_hint_y:1
    			pos_hint:{"left":0.9,"top":0.9}
    			Image:  
    				source: "graphs/ST_logo.png"
    				x:self.parent.x
    				y:self.parent.y
    				size: self.parent.width, self.parent.height 
    				allow_stretch: True
    		    	keep_ratio: True

			#Title
            Label:  
    			id: front_title
			
    			text: "SPIN TUNNELING"
    			font_size:72
    			font_name:'07558_CenturyGothic' 
    			bold: True
    			color:0, 0, 0, 1
    			multiline: True
			
    			pos_hint: {"center_x":0.5,"center_y":0.5}
    			#diccionary to assign pos:{"x","y","top","bottom","left","right"}
			
    			size_hint_x: 0.6
    			size_hint_y:0.1 
    			halign:"center"
    			valign:"middle" #centered paragraph sentences
            
            #Text
            Label:  
    			id: welcome_text
    			
    			text: "WELCOME TO SPIN TUNNELING, THE EXPERIENCE YOU MUST GO THROUGH." 
    			font_size:28
    			font_name:'39335_UniversCondensed' 
    			bold: False
    			color:0, 0, 0, 1
    			multiline: True
			
    			pos_hint: {"center_x":0.5,"center_y":0.15}			
    			size_hint_x: 0.6
    			size_hint_y:0.1 
    			halign:"center"
    			valign:"middle" #centered paragraph sentences
            
            BoxLayout: #for languages
    			size_hint_x: 0.15
    			size_hint_y:0.3
    			pos_hint:{"right":0.98,"top":0.9}
    			orientation: "horizontal"
			
    			#add buttons for languages
			
    			#CATALAN
    			Button: 
    	            id: button_cat
    	            text: "CAT"
    	            font_name:'39335_UniversCondensed'
    	            color: 23/255, 190/255, 255/255, 1
    	        	font_size:18
    				bold: True
                	halign:"center"
    				valign:"middle"
    				on_release: #root.catalan() 
				
                #ESP
    			Button:  
    	            id: button_esp
    	            text: "CAST"
    	            font_name:'39335_UniversCondensed'
    	            color: 23/255, 190/255, 255/255, 1
    	        	font_size:18
    				bold: True
                	halign:"center"
    				valign:"middle"
    				on_release: #root.esp()
				
    			#ENGLISH	  
    			Button:  
    	            id: button_eng
    	            text: "ENG"
    	            font_name:'39335_UniversCondensed'
    	            color: 23/255, 190/255, 255/255, 1
    	        	font_size:18
                    background_color: (0,0,1,0.8)
    				bold: True
                	halign:"center"
    				valign:"middle"
    				on_release: #root.english()
				
        #MIDDLE SCREEN
        FloatLayout: 
			size_hint_x: 1 
			size_hint_y:0.5
			pos_hint:{"x":0,"y":0.25}
			
            #Text
            Label:  
    			id: front_text
    			
    			text: "Instructions \n or \n explanation (e.g. information btns will contain the physical explanation.)" 
    			font_size:28
    			font_name:'39335_UniversCondensed' 
    			bold: False
    			color:0, 0, 0, 1
    			multiline: True
    			
    			pos_hint: {"left":0.9,"top":0.9}			
    			size_hint_x: 0.8
    			size_hint_y:0.2 
    			halign:"left"
    			valign: "top"


            Button:
                id: btn_reso
                
                text: "RESONANCE"
                font_size:44
                font_name:'39335_UniversCondensed'
                bold: False
    			color: 23/255, 190/255, 255/255, 1
    			
    			pos_hint: {"center_x":0.35,"center_y":0.5}
    			size_hint_x: 0.25
    			size_hint_y: 0.15
                halign:"center"
    			valign:"middle"
    			
    			on_press:
        			root.manager.get_screen('Experiment').ids.front_title.text = "RESONANCE"
        			root.manager.get_screen('Experiment').screen_decider = 1
                    root.manager.current = 'Experiment'
                

            Button:
                id: btn_exp
                
                text: "EXPERIMENT"
                font_size:44
                font_name:'39335_UniversCondensed'
                bold: False
    			color: 23/255, 190/255, 255/255, 1
    			
    			pos_hint: {"center_x":0.35,"center_y":0.3}
    			size_hint_x: 0.25
    			size_hint_y: 0.15
                halign:"center"
    			valign:"middle"
    			
                on_press:
                    root.manager.get_screen('Experiment').ids.front_title.text = "EXPERIMENT"
        			root.manager.get_screen('Experiment').screen_decider = 2
                    root.manager.current = 'Experiment'
                    
            Button:
                id: btn_tut
                
                text: "TUTORIAL" if btn_tut.state == "normal" else "Sth Incoming"
                font_size:44
                font_name:'39335_UniversCondensed'
                bold: False
    			color: 23/255, 190/255, 255/255, 1
    			
    			pos_hint: {"center_x":0.65,"center_y":0.5}
    			size_hint_x: 0.25
    			size_hint_y: 0.15
                halign:"center"
    			valign:"middle"
    			
    		Button:
                id: btn_game
                
                text: "GAME" if btn_game.state == "normal" else "Sth Incoming"
                font_size:44
                font_name:'39335_UniversCondensed'
                bold: False
    			color: 23/255, 190/255, 255/255, 1
    			
    			pos_hint: {"center_x":0.65,"center_y":0.3}
    			size_hint_x: 0.25
    			size_hint_y: 0.15
                halign:"center"
    			valign:"middle"
         
        #Bottom SCREEN
        FloatLayout: 
			size_hint_x: 1 
			size_hint_y:0.25
			pos_hint:{"x":0,"y":0}
			
			GridLayout:
    			cols: 1
    			rows: 2
    			
    			#Tutors
    			Label
        			id: tutors_text
        			
        			text: "Tutors: \n Dr. Carles Calero & Dr. Bruno Julia" #Julià Cant plot accents
        			font_size:28
        			font_name:'39335_UniversCondensed' 
        			bold: False
        			color:0, 0, 0, 1
        			multiline: True
        			
        			pos_hint: {"x":0,"y":0}			
        			size_hint_x: 0.4
        			size_hint_y:1
        			halign:"left"
        			valign: "top"
    			 
                #Institutional Logos
                GridLayout:
        			cols: 3
        			rows: 1 
        			
        			#Nano logo
        			Image:  
        				source: "graphs/logo_nano.jpg"
        				x:self.parent.x
        				y:self.parent.y
        				size: self.parent.width, self.parent.height 
            			allow_stretch: True
        		    	keep_ratio: True 
                     
                #UB
        			Image: #iccub logo 
        				source: "graphs/ub_logo.png"
        				x:self.parent.x
        				y:self.parent.y
        				size: self.parent.width, self.parent.height 
        				allow_stretch: True
        		    	keep_ratio: True
    		    	
    		    	
                #ICCUB   			
        			Image: #iccub logo 
        				source: "graphs/logoiccub.png"
        				x:self.parent.x
        				y:self.parent.y
        				size: self.parent.width, self.parent.height 
        				allow_stretch: True
        		    	keep_ratio: True

<Experiment_Screen>
    name:'Experiment'
    
    graphic_box1:graphic_box1
    graphic_box2:graphic_box2
    play_btn:play_btn
    
    loading_text:loading_text
    
    #Experiment LAYOUT
    FloatLayout: 
		id: experiment_layout
		
		#size: root.width, root.height #layout that occupies all image
		size_hint_x:1 
		size_hint_y:1
    	
    	#Sets bg color
    	canvas.before: 
			Color: 
				rgba: (1,1,1,1) 
			Rectangle: 
				size: self.size 
				pos: self.pos
				
        #TOP SCREEN
        FloatLayout: 
			size_hint_x: 1 
			size_hint_y:0.2
			pos_hint:{"x":0,"y":0.8}
			
			#Information PopUp
            Button:
                id: popup_exp
                
                text: "i"
                font_size:44
                font_name:'00803_AbrazoScriptSSiBold'
                bold: True
    			color: 1, 1, 1, 1
    			background_color: 0, 0, 0, 1
    			
    			pos_hint: {"center_x":0.05,"center_y":0.6}
    			size_hint_x: 0.05
    			size_hint_y: 0.4
                halign:"center"
    			valign:"middle"
    			on_release:
        			root.popup_btn()

			#Title
            Label:  
    			id: front_title
			
    			text: "EXPERIMENT"
    			font_size:60
    			font_name:'07558_CenturyGothic' 
    			bold: True
    			color:0, 0, 0, 1
    			multiline: True
			
    			pos_hint: {"center_x":0.25,"center_y":0.6}
    			#diccionary to assign pos:{"x","y","top","bottom","left","right"}
			
    			size_hint_x: 0.6
    			size_hint_y:0.1 
    			halign:"center"
    			valign:"middle" #centered paragraph sentences
            
            #Text
            Label:  
    			id: explanation_text
    			
    			text: "Feel free to try whatever you want and analyse what is happening." 
    			font_size:20
    			font_name:'39335_UniversCondensed' 
    			bold: False
    			color:0, 0, 0, 1
    			multiline: True
			
    			pos_hint: {"center_x":0.6,"center_y":0.45}			
    			size_hint_x: 0.6
    			size_hint_y:0.1 
    			halign:"center"
    			valign:"middle" #centered paragraph sentences
            
            BoxLayout: #for languages
    			size_hint_x: 0.15
    			size_hint_y:0.3
    			pos_hint:{"right":0.98,"top":0.9}
    			orientation: "horizontal"
			
    			#add buttons for languages
			
    			#CATALAN
    			Button: 
    	            id: button_cat
    	            text: "CAT"
    	            font_name:'39335_UniversCondensed'
    	            color: 23/255, 190/255, 255/255, 1
    	        	font_size:18
    				bold: True
                	halign:"center"
    				valign:"middle" #lletra centrada
    				on_release: #root.catalan() 
				
                #ESP
    			Button:  
    	            id: button_esp
    	            text: "CAST"
    	            font_name:'39335_UniversCondensed'
    	            color: 23/255, 190/255, 255/255, 1
    	        	font_size:18
    				bold: True
                	halign:"center"
    				valign:"middle" #lletra centrada
    				on_release: #root.esp()
				
    			#ENGLISH	  
    			Button:  
    	            id: button_eng
    	            text: "ENG"
    	            font_name:'39335_UniversCondensed'
    	            color: 23/255, 190/255, 255/255, 1
    	        	font_size:18
                    background_color: (0,0,1,0.8)
    				bold: True
                	halign:"center"
    				valign:"middle" #lletra centrada
    				on_release: #root.english()
    				

        #MIDDLE SCREEN 1
        FloatLayout: 
			size_hint_x: 0.6 
			size_hint_y:0.07
			pos_hint:{"x":0.1,"y":0.78}
			
			#Sets bg color
        	canvas.before: 
    			Color: 
    				rgba: (0, 0.08, 0.31, 1) 
    			Rectangle: 
    				size: self.size 
    				pos: self.pos
    				
    				
            #HAMILTONIANS Labels
            #HAM 1
            Label:
        		size_hint:0.4,0.9
        		pos_hint: {'x':0.05,'center_y':0.5}
        		
        		id: H1
                  
                text: " "

          
            GridLayout:
    			cols:1 
    			rows:1
    			
    			size_hint:0.4,0.9
            	pos_hint: {'x':0.05,'center_y':0.5}
            
                Image: #we put an image
    				source: "graphs/Ham_1.png"
    	    		allow_stretch: True
    	    		keep_ratio: True
    				
    		#HAM 2
            Button:
        		size_hint:0.3,0.9
        		pos_hint: {'x':0.45,'center_y':0.5}
        		
        		id: H2_btn
                    
                text: " "

                disabled: True
                  
                halign:"center"
                valign:"middle"
    
                on_release:
                    root.ham=2
                    H2_btn.disabled=True
                    H3_btn.disabled=False
                    B_change.text='B = {:.2f}'.format(B_slider.value)
        		
            
            GridLayout:
    			cols:1 
    			rows:1
    			
    			size_hint:0.3,0.9
        		pos_hint: {'x':0.45,'center_y':0.5}
            
                Image: #we put an image
    				source: "graphs/Ham_2.png"
    	    		allow_stretch: True
    	    		keep_ratio: True
    	    
    	    #HAM 3
            Button:
        		size_hint:0.2,0.9
        		pos_hint: {'x':0.75,'center_y':0.5}
        		
        		id: H3_btn
                    
                text: " "

                disabled: False
                  
                halign:"center"
                valign:"middle"
    
                on_release:
                    root.ham=3
                    H2_btn.disabled=False
                    H3_btn.disabled=True
                    B_change.text='hx = {:.2f}'.format(B_slider.value)
      		
            GridLayout:
    			cols:1 
    			rows:1
    			
    			size_hint:0.2,0.7
        		pos_hint: {'x':0.75,'center_y':0.5}
            
                Image: #we put an image
    				source: "graphs/Ham_3.png"
    	    		allow_stretch: True
    	    		keep_ratio: True

        
        #MIDDLE SCREEN 2
        FloatLayout: 
			size_hint_x: 0.18 
			size_hint_y:0.07
			pos_hint:{"x":0.72,"y":0.78}
			
			#Sets bg color
        	canvas.before: 
    			Color: 
    				rgba: (0, 0.08, 0.31, 1) 
    			Rectangle: 
    				size: self.size 
    				pos: self.pos
            
            
            #SPIN VALUE
            BoxLayout:
                id:spin_box
        
                orientation:'horizontal'
                size_hint:0.6,0.8
                pos_hint:{'center_x':0.5, 'center_y': 0.5}
                
                #Text
                Label:
        			id: spin_text
        			
        			text: "Spin = " 
        			font_size:20
        			font_name:'39335_UniversCondensed' 
        			bold: True
        			color:1, 1, 1, 1

        			halign:"center"
        			valign:"middle" #centered paragraph sentences
            
                
                Spinner:
                    id: spinner_s
                    
                    # initially text on spinner
                    text: "1"
                    # total values on spinner
                    values: ["1", "2", "3", "4", "5", "6"]
                    
                    # Callback 
                    on_text:
                        root.spinner_clicked_s(spinner_s.text)
          
                
  
                # declaring size of the spinner
                # and the position of it
                #size_hint: 0.05, 0.4
                
    	    		
        #MIDDLE SCREEN 3
        FloatLayout: 
			size_hint_x: 0.8
			size_hint_y:0.12
			pos_hint:{"x":0.1,"y":0.65}
			
			#Sets bg color
        	canvas.before: 
    			Color: 
    				rgba: (0, 0.08, 0.31, 1) 
    			Rectangle: 
    				size: self.size 
    				pos: self.pos
        
            #SLIDERS
            GridLayout:
    			cols: 5
    			rows: 2 
        		size_hint: 0.85,0.8
        		pos_hint: {'x':0.03,'center_y':0.5}
        
        
                #LABELS
                #t0
                Label:
                    text: 't0 = {:.0f}'.format(t0_slider.value)
                    font_size:18
                    font_name:'39335_UniversCondensed'
        			bold: True
                    
                    
                #tf
                Label:
                    text: 'tf = {:.0f}'.format(tf_slider.value)
                    font_size:18
                    font_name:'39335_UniversCondensed' 
        			bold: True
                
                #D
                Label:
                    text: 'D = {:.1f}'.format(D_slider.value)
                    font_size:18
                    font_name:'39335_UniversCondensed' 
        			bold: True
                
                #hz
                Label:
                    text: 'hz = {:.2f}'.format(hz_slider.value)
                    font_size:18
                    font_name:'39335_UniversCondensed' 
        			bold: True
                #B
                Label:
                    id: B_change
                    text: 'B = {:.2f}'.format(B_slider.value)
                    font_size:18
                    font_name:'39335_UniversCondensed' 
        			bold: True
                
                
                #SLIDERS
                #t0
                Slider:
                    id:t0_slider

                    value: -35
                    min: -50
                    max: 0
                    step: 2
                    
                    on_value:
                        root.t0=t0_slider.value
                
                #tf
                Slider:
                    id:tf_slider
                    
                    value: 35
                    min: 0
                    max: 50
                    step: 2
                    
                    on_value:
                        root.tf=tf_slider.value
                
                #D
                Slider:
                    id:D_slider
                    
                    value: 5
                    min: 1
                    max: 10
                    step: 0.5
                    
                    on_value:
                        root.D=D_slider.value
                
                #hz        
                Slider:
                    id:hz_slider
                    
                    value: 1
                    min: 0.1
                    max: 3
                    step: 0.05
                    
                    on_value:
                        root.hz=hz_slider.value
                
                #B
                Slider:
                    id:B_slider
                    
                    value: 1
                    min: 0.1
                    max: 3
                    step: 0.05
                    
                    on_value:
                        root.B=B_slider.value


            Button:
                id: send_btn
                
                text: "SEND"
                font_size:32
                font_name:'39335_UniversCondensed'
                bold: False
        		color: 23/255, 190/255, 255/255, 1
        
                pos_hint: {"right":0.98,"center_y":0.5}
                size_hint_x: 0.1
                size_hint_y: 0.55
                halign:"center"
                valign:"middle"
                
                on_press:
                    loading_text.text= 'LOADING...'
                    
                on_release:
                    root.send()
                    loading_text.text= 'DONE'
                    root.done_anim()
                    play_btn.disabled = False  

    # GRAFICS
    GridLayout:
        rows: 1
        cols: 2
        size_hint:1,0.4
        pos_hint: {'center_x':0.5,'y':0.25}
        
        BoxLayout:
            id:graphic_box1
            #orientation:'vertical'
        BoxLayout:
            id:graphic_box2
            #orientation:'vertical'
    
    
    
    #DEMOS SQUARE
    GridLayout:
        rows: 2
        cols: 3
        
		size_hint_x: 0.25 
		size_hint_y:0.15
		pos_hint:{"x":0.15,"y":0.05}
		
		
		#DEMO 1	
		Button:
            id: demo1
                
            text: "DEMO S=1"
            font_size:14
            font_name:'07558_CenturyGothic'
            bold: True

    		on_press:
                loading_text.text= 'LOADING...'
                
            on_release:
                spinner_s.text='1'
                root.spinner_clicked_s(spinner_s.text)
            
                D_slider.value=4
        		hz_slider.value=1.2
        		B_slider.value=1.8
        		
                root.send()
                loading_text.text= 'DONE'
                root.done_anim()
                play_btn.disabled = False
		
		#DEMO 2	
		Button:
            id: demo2
                
            text: "DEMO S=2"
            font_size:14
            font_name:'07558_CenturyGothic'
            bold: True

    		on_press:
                loading_text.text= 'LOADING...'
                
            on_release:
                spinner_s.text='2'
                root.spinner_clicked_s(spinner_s.text)
            
                D_slider.value=4
        		hz_slider.value=1.2
        		B_slider.value=1.8
        		
                root.send()
                loading_text.text= 'DONE'
                root.done_anim()
                play_btn.disabled = False
		
		#DEMO 3
        Button:
            id: demo3
                
            text: "DEMO S=3"
            font_size:14
            font_name:'07558_CenturyGothic'
            bold: True

    		on_press:
                loading_text.text= 'LOADING...'
                
            on_release:
                spinner_s.text='3'
                root.spinner_clicked_s(spinner_s.text)
            
                D_slider.value=4
        		hz_slider.value=1.2
        		B_slider.value=1.8
        		
                root.send()
                loading_text.text= 'DONE'
                root.done_anim()
                play_btn.disabled = False
                
        #DEMO 4
		Button:
            id: demo4
                
            text: "DEMO S=4"
            font_size:14
            font_name:'07558_CenturyGothic'
            bold: True

    		on_press:
                loading_text.text= 'LOADING...'
                
            on_release:
                spinner_s.text='4'
                root.spinner_clicked_s(spinner_s.text)
            
                D_slider.value=4
        		hz_slider.value=1.2
        		B_slider.value=1.8
        		
                root.send()
                loading_text.text= 'DONE'
                root.done_anim()
                play_btn.disabled = False
        
        #DEMO 5	
		Button:
            id: demo5
                
            text: "DEMO S=5"
            font_size:14
            font_name:'07558_CenturyGothic'
            bold: True

    		on_press:
                loading_text.text= 'LOADING...'
                
            on_release:
                spinner_s.text='5'
                root.spinner_clicked_s(spinner_s.text)
            
                D_slider.value=4
        		hz_slider.value=1.2
        		B_slider.value=1.8
        		
                root.send()
                loading_text.text= 'DONE'
                root.done_anim()
                play_btn.disabled = False
        
        #DEMO 6	
		Button:
            id: demo6
                
            text: "DEMO S=6"
            font_size:14
            font_name:'07558_CenturyGothic'
            bold: True

    		on_press:
                loading_text.text= 'LOADING...'
                
            on_release:
                spinner_s.text='6'
                root.spinner_clicked_s(spinner_s.text)
            
                D_slider.value=4
        		hz_slider.value=1.2
        		B_slider.value=1.8
        		
                root.send()
                loading_text.text= 'DONE'
                root.done_anim()
                play_btn.disabled = False
        
    #LOADING
    FloatLayout: 
		size_hint_x: 0.27 
		size_hint_y:0.15
		pos_hint:{"x":0.48,"y":0.05}
			
		#Sets bg color
        canvas.before: 
    		Color: 
    			rgba: (0, 0, 0, 1) 
    		Rectangle: 
    			size: self.size 
    			pos: self.pos
        Label:
            id:loading_text
                
            text: 'WAITING'
            size_hint: 0.6,0.8
            pos_hint: {'x':0.05,'center_y':0.5}
            
            font_size:40
        	font_name:'07558_CenturyGothic' 
        	bold: False
        	color:1, 1, 1, 1
        	multiline: True
    GridLayout:
        cols: 1
    	rows: 1 
        size_hint: 0.04,0.02
        pos_hint: {'center_x':0.7,'center_y':0.12}
        
        #Loading rotating
        FloatLayout:
            canvas.before:
                PushMatrix
                Rotate:
                    angle: root.angle
                    axis: 0, 0, 1
                    origin: self.center
            canvas.after:
                PopMatrix
                

            Image:
                size_hint: 0.2, 0.2
                pos_hint: {'center_x': 0.9, 'center_y': 0.9}

    GridLayout:
        cols: 1
        rows: 3
        
        size_hint_x: 0.1 
		size_hint_y:0.15
		pos_hint:{"x":0.8,"y":0.05}
    
        # PLAY
        Button:
            id: play_btn
            
            disabled: True
            
            text:'PLAY'
            font_size: 24
            font_name:'07558_CenturyGothic'
            bold: True
            
            halign:"center"
        	valign:"middle"
            
            on_press:
                root.play()
                play_btn.disabled = False
                pause_btn.disabled = False
                reset_btn.disabled = False
    
        # PAUSE
        Button:
            id: pause_btn
            
            disabled: True
            
            text:'PAUSE'
            font_size: 24
            font_name:'07558_CenturyGothic'
            bold: True
    		
            halign:"center"
        	valign:"middle"
            
            on_press:
                root.pause()
                play_btn.disabled = False
                pause_btn.disabled = True
                reset_btn.disabled = False
                
        # RESET
        Button:
            id: reset_btn
            
            disabled: True
            
            text:'RESET'
            font_size: 24
            font_name:'07558_CenturyGothic'
            bold: True
    		
            halign:"center"
        	valign:"middle"
            
            on_press:
                root.reset()
                play_btn.disabled = False
                pause_btn.disabled = True
                reset_btn.disabled = True

    GridLayout:
        cols: 1
        rows: 3
        
        size_hint_x: 0.05 
		size_hint_y:0.15
		pos_hint:{"x":0.92,"y":0.05}
    
        # x1
        Button:
            id: x1_btn
            
            disabled: True
            
            text:'x1'
            font_size: 24
            font_name:'07558_CenturyGothic'
            bold: True
            
            halign:"center"
        	valign:"middle"
            
            on_press:
                root.vel_decider=1
                x1_btn.disabled = True
                x2_btn.disabled = False
                x4_btn.disabled = False
    
        # x2
        Button:
            id: x2_btn
            
            text:'x2'
            font_size: 24
            font_name:'07558_CenturyGothic'
            bold: True
    		
            halign:"center"
        	valign:"middle"
            
            on_press:
                root.vel_decider=2
                x1_btn.disabled = False
                x2_btn.disabled = True
                x4_btn.disabled = False
                
        # RESET
        Button:
            id: x4_btn
            
            text:'x4'
            font_size: 24
            font_name:'07558_CenturyGothic'
            bold: True
    		
            halign:"center"
        	valign:"middle"
            
            on_press:
                root.vel_decider=4
                x1_btn.disabled = False
                x2_btn.disabled = False
                x4_btn.disabled = True

    # BACK
    Button:
        id: back_btn
        
        text:'Back'
        font_size:44
        font_name:'00803_AbrazoScriptSSiBold'
        bold: True
    	color: 1, 1, 1, 1
    	background_color: 0, 0, 0, 1
        
        pos_hint:{'left':0.8,'bottom':0.8}
        size_hint:0.1,0.1
        halign:"center"
    	valign:"middle"
        
        on_press:
            play_btn.disabled = True
            pause_btn.disabled = True
            reset_btn.disabled = True
            root.pause()
            root.count=0
            root.back()
            root.manager.current = 'Menu'