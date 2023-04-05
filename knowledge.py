Pascal_part_connection = {
    "aeroplane":[('aeroplane_body','aeroplane_lwing'),('aeroplane_body','aeroplane_stern'),('aeroplane_body','aeroplane_rwing')],
    "bird":[("bird_torso","bird_head"),("bird_torso","bird_wing"),("bird_torso","bird_leg")],
    "bus":[("bus_frontside","bus_lrside")],
    "car":[("car_frontside","car_rlside"),("car_rlside","car_backside")],
    "cat":[("cat_torso","cat_head"),("cat_torso","cat_bleg"),("cat_torso","cat_fleg")],
    "dog":[("dog_torso","dog_head"),("dog_torso","dog_fleg"),("dog_torso","dog_bleg")],
    "horse":[("horse_torso","horse_head"),("horse_torso","horse_leg"),("horse_torso","horse_tail")],
    "person":[("person_torso","person_head"),("person_torso","person_larm"),("person_torso","person_rarm"),("person_torso","person_rleg"),("person_torso","person_lleg")],
    "sheep":[("sheep_torso","sheep_head"),("sheep_torso","sheep_leg")],
    "train":[("train_coach","train_head")],
}

PartImageNet_connection = {
    "Quadruped":[('Quadruped Head','Quadruped Body'),('Quadruped Foot','Quadruped Body'),('Quadruped Tail','Quadruped Body')],
    "Biped": [('Biped Head','Biped Body'),('Biped Body','Biped Hand'),('Biped Foot','Biped Body'),('Biped Tail', 'Biped Body')],
    "Fish": [('Fish Head', 'Fish Body'), ('Fish Body', 'Fish Fin'), ('Fish Tail', 'Fish Body')],
    "Bird": [('Bird Head', 'Bird Body'), ('Bird Wing', 'Bird Body'), ('Bird Body', 'Bird Tail'), ('Bird Body', 'Bird Foot')],
    "Snake": [('Snake Head', 'Snake Body')],
    "Reptile": [('Reptile Head', 'Reptile Body'), ('Reptile Foot', 'Reptile Body'), ('Reptile Tail', 'Reptile Body')],
    "Bicycle": [('Bicycle Body', 'Bicycle Head'), ('Bicycle Seat', 'Bicycle Body'), ('Bicycle Tier', 'Bicycle Body')],
    "Aeroplane": [('Aeroplane Head', 'Aeroplane Body'), ('Aeroplane Body', 'Aeroplane Engine'), ('Aeroplane Wing','Aeroplane Body'), ('Aeroplane Tail', 'Aeroplane Body')],
}

FaceBody_connection = {
    'head':[('head_nose','head_skin'),('head_mouth','head_skin'),('head_l_eye','head_skin'),('head_r_eye','head_skin'),('head_skin','head_neck'),('head_hair','head_skin'),('head_ear','head_skin')],
    'body':[('body_cloth','body_left-arm'),('body_cloth','body_left-leg'),('body_cloth','body_right-arm'),('body_cloth','body_right-leg')]
}