# CHAPTER ONE

## INTRODUCTION

### 1.1 Background of the Study

Robotics has moved beyond isolated industrial automation toward shared human-centered environments such as factories, hospitals, homes, offices, and service spaces. In these environments, robots are expected to operate safely and intelligently around people rather than simply execute fixed commands in isolation. This shift has made Human-Robot Interaction (HRI) a major area of interest in robotics research. For robots to function effectively in human spaces, they must be able to interpret human behavior and respond in a way that is safe, timely, and natural.

One important aspect of this challenge is human intention recognition. Human intention recognition refers to the process of inferring a person's likely action, command, or movement goal from observable cues. In everyday interaction, people communicate intention not only through explicit verbal or manual commands, but also through hand gestures, body posture, and movement direction. For example, a raised hand may indicate a stop command, a directional hand gesture may indicate a navigation instruction, and body motion toward a robot may imply a need for caution or avoidance. A robot that can recognize such cues can behave more intelligently than one that responds only after a complete command has been issued.

Recent developments in computer vision and machine learning have made vision-based intention recognition more practical. Using a monocular camera and pose-estimation tools such as MediaPipe, it is now possible to extract both hand landmarks and full-body pose landmarks in real time without requiring users to wear sensors. These landmarks can then be analyzed using machine learning models to classify gestures and movement patterns. Sequential learning models such as Long Short-Term Memory (LSTM) networks and graph-based models such as Spatial-Temporal Graph Convolutional Networks (ST-GCN) are especially useful for learning temporal human motion patterns from skeleton sequences.

Despite these advances, reliable real-time intention recognition remains difficult. A system may perform well in offline evaluation yet behave inconsistently in live use due to landmark noise, label noise, frame-to-frame instability, and environmental variation. In particular, distinguishing between approaching and moving away from a single camera stream can be unstable if the system relies only on learned class probabilities. This creates practical problems for robot response because a robot should react differently when a person is moving toward it, moving away from it, standing still, or issuing a direct gesture command.

This study addresses that challenge by developing a camera-based robotic application for human intention recognition using both hand gesture and body motion analysis. The system combines three perception pathways: a hand-gesture classifier for explicit commands, an LSTM-based movement classifier for body motion, and an ST-GCN-based skeleton action model as a secondary motion signal. In addition, the system incorporates interpretable front/back reasoning based on signed pose depth change and apparent body-scale change over time. The output of these modules is integrated into a ROS2-compatible decision pipeline that generates appropriate navigation-related robot responses.

The resulting system provides a practical example of hybrid human intention recognition, where learned models are combined with physically interpretable motion cues to improve live reliability. This contributes to safer and more responsive human-robot interaction in environments where explicit commands and implicit motion cues must both be understood.

### 1.2 Problem Statement

Many robotic systems still operate in a reactive manner, relying on predefined commands or simple one-to-one mappings between inputs and robot actions. While such systems may function adequately in controlled settings, they are limited in shared environments where humans communicate intention continuously through movement and gesture. A robot that cannot interpret these cues reliably may respond too late, respond incorrectly, or fail to distinguish between a direct command and a contextual movement pattern.

Although gesture recognition and motion recognition have been widely studied, they are often treated as separate tasks rather than integrated parts of one intention-recognition system. Some systems focus only on hand gestures, while others focus only on human trajectory or skeleton motion. In addition, many published approaches are evaluated under controlled conditions and do not sufficiently address the difference between offline performance and live runtime behavior. As a result, a model can appear accurate during training and testing while still producing unstable or contradictory results in real-time operation.

This problem became evident during the development of this project. Initial motion models could classify movement categories offline, but front/back classes such as approaching and moving_away were less reliable in live camera operation than expected. False stationary transitions, delayed switching, and contradictory front/back predictions reduced the quality of robot response. This showed that a robust intention-recognition system requires more than a standard classifier pipeline; it must also handle temporal instability, dataset noise, and live camera variability.

The core problem addressed in this study is therefore the absence of a robust, integrated, real-time framework that combines hand gesture recognition and body motion understanding into a meaningful interpretation of human intention for robot response. The system must detect explicit gesture commands, classify contextual body motion, distinguish reliably between approaching and moving away, and generate safe robot actions in a live and ROS2-integrated setting.

### 1.3 Objectives of the Study

#### 1.3.1 General Objective

The general objective of this study is to design and implement a camera-based robotic system capable of recognizing human intention from hand gestures and body motion and using that information to drive safe robot response behavior.

#### 1.3.2 Specific Objectives

To achieve the general objective, the following specific objectives were pursued:

1. To review relevant literature on human intention recognition, gesture recognition, pose-based motion analysis, and robot behavior in shared environments.
2. To develop a vision-based perception pipeline for extracting hand and body landmarks from live video using MediaPipe.
3. To design and implement a gesture-recognition module for explicit command classification.
4. To design and implement sequence-based motion models for classifying approaching, moving_away, moving_left, moving_right, and stationary behavior.
5. To improve the reliability of front/back motion recognition by incorporating signed pose-depth and body-scale reasoning into live inference.
6. To integrate the perception modules into a ROS2-compatible robot decision pipeline.
7. To evaluate the system in both direct live-camera mode and ROS2/Gazebo-based operation.

### 1.4 Significance of the Study

This study is significant because it addresses a practical requirement in modern human-robot interaction: enabling robots to understand natural human cues rather than relying only on explicit command structures. By integrating gesture recognition with body-motion understanding, the work moves toward a more realistic interaction model in which the robot can react to both direct user commands and contextual human behavior.

From an academic perspective, the study contributes an implementation-driven framework for multimodal intention recognition using a monocular camera, hand landmarks, body pose sequences, machine learning models, and ROS2 integration. A key contribution is the introduction of direction-aware reasoning for front/back movement based on signed depth and body-scale trends. This hybrid design addresses a common gap between offline model accuracy and real-time behavioral reliability.

From a practical perspective, the system is relevant to collaborative robotics, assistive robotics, and service robotics. In such applications, a robot should respond differently if a user commands it with a gesture, walks toward it, moves laterally across its path, or moves away from it. A system that can reliably interpret these situations contributes to safer, more intuitive, and more effective interaction.

### 1.5 Brief Methodology

This project was implemented through a structured workflow involving data preparation, model development, live inference design, and ROS2 integration.

First, a camera-based perception pipeline was built using MediaPipe to extract hand landmarks and full-body pose landmarks in real time. The hand landmarks were used as input to a landmark-based gesture classifier. The body pose landmarks were collected into temporal sequences and processed for movement recognition.

Second, sequence-based machine learning models were developed. An LSTM model was trained on pose-derived temporal movement features, while an ST-GCN model was trained on skeleton sequences to provide a secondary action-recognition pathway. These models were used to classify body movement into five classes: approaching, moving_away, moving_left, moving_right, and stationary.

Third, the recorded movement dataset was audited, particularly the forward/backward classes. Signed pose-depth and apparent body-scale trends were used to identify likely label inconsistencies in approaching and moving_away samples. A cleaned movement dataset was then created by retaining only high-confidence agreeing forward/backward samples and retraining the motion models on that improved data.

Fourth, live inference logic was improved to better match runtime behavior. The final system uses signed depth and body-scale change as strong direction cues for front/back motion. When the direction signal is clear, it is used to stabilize or override contradictory model output between approaching and moving_away. This reduces false stationary transitions and improves switching during live reversals.

Finally, the system was integrated into a ROS2-compatible robotic application. The perception outputs were converted into robot-response decisions, and the complete system was tested both in desktop live mode and in a ROS2/Gazebo environment.

### 1.6 Scope of the Study

This study focuses on the design and implementation of a camera-based intention-recognition system for robot interaction using hand gestures and body motion. The system uses a monocular camera, MediaPipe landmark extraction, machine learning models for gesture and motion recognition, and a ROS2-based robot decision pipeline.

The gesture component is limited to a small set of explicit commands: stop, forward, backward, left, and right. The motion component is limited to five movement classes: approaching, moving_away, moving_left, moving_right, and stationary. The robot behavior implemented in this project is intentionally conservative: gesture commands have the highest priority, approaching triggers avoidance, lateral motion triggers directional adjustment, and moving_away or stationary results in a hold state.

The work includes desktop live testing and ROS2/Gazebo robot testing. It does not include natural-language interaction, multi-person tracking, semantic environment understanding, reinforcement-learning-based navigation, or large-scale autonomous planning. The main scope is reliable real-time perception and intention-aware robot response in a controlled experimental setting.

### 1.7 Organization of the Study

This work is organized into three chapters. Chapter One presents the background of the study, the problem statement, objectives, significance, methodology, and scope. Chapter Two reviews relevant literature on human intention recognition, gesture recognition, pose-based motion understanding, and robot interaction systems. Chapter Three describes the methodology and implementation framework used to develop, train, integrate, and evaluate the final system.

# CHAPTER TWO

## LITERATURE REVIEW

### 2.0 Overview

This chapter reviews relevant studies related to human intention recognition, gesture recognition, pose-based motion analysis, and robot response in shared environments. Existing research shows that natural human cues such as hand gestures and body movement can be used to improve robot awareness and interaction quality. However, many systems focus on only one modality, rely on specialized hardware, or do not sufficiently address the gap between offline recognition accuracy and real-time runtime reliability. This study builds on those works by combining gesture recognition, movement understanding, skeleton-based action modeling, and live direction-aware reasoning in a single camera-based robotic application.

### 2.1 Theoretical Review

This study is grounded in Human-Robot Interaction theory, human intention theory, computer vision-based pose estimation, temporal motion modeling, and multimodal fusion.

Human intention theory assumes that human actions are goal-directed and that observable cues often contain predictive information before a task is completed. In HRI, this supports the idea that a robot should not merely react after a command is fully executed; rather, it should interpret ongoing human behavior early enough to respond safely and naturally.

Gesture recognition theory supports the use of structured hand landmarks as meaningful indicators of explicit human commands. A gesture can be understood as a deliberate visual signal that carries symbolic meaning. When hand landmarks are tracked consistently, machine learning models can classify such gestures into robot-action categories.

Pose-based motion understanding assumes that body motion over time contains sufficient information to infer movement state and short-term intention. Temporal models such as LSTM networks are effective when motion can be represented as ordered sequences of landmark-derived features. Graph-based models such as ST-GCN are effective when the human body is treated as a connected skeleton graph in which joints are nodes and anatomical relationships are edges. These models capture both spatial relationships and temporal evolution.

Multimodal fusion theory suggests that combining more than one information source can improve robustness and reduce ambiguity. In this work, explicit hand gestures provide direct commands, while body motion provides contextual understanding of the human state. The system also combines learned sequence-model output with physically interpretable cues derived from signed depth and apparent body-scale change. This hybrid strategy improves reliability when raw model predictions alone are unstable.

Finally, safe robot response theory in shared spaces supports the use of conservative motion policies. In environments where humans and robots coexist, it is preferable for the robot to stop or avoid when uncertainty or human proximity increases, rather than rely on aggressive autonomy. This principle influenced the final response logic of the system.

### 2.2 Review of Related Works

#### 2.2.1 Review One

##### 2.2.1.1 Research Topic and Author

Real-time Feasibility of a Human Intention Method Evaluated Through a Competitive Human-Robot Reaching Game (Tsitos et al., 2022)

##### 2.2.1.2 Objectives

The objective of the study was to investigate whether a robot could predict human reaching intention early enough for real-time response using visual pose data extracted from a single RGB-D sensor. The work sought to determine whether early intention prediction was practically useful in human-robot competition tasks.

##### 2.2.1.3 Methodology

The researchers used an RGB-D camera and OpenPose to estimate human pose, particularly wrist position, during reaching tasks. Temporal features were extracted and used to train standard machine-learning classifiers such as Logistic Regression, Support Vector Machine, Decision Tree, and Naive Bayes. The intention-prediction module was integrated into a ROS pipeline connected to a UR3 robot and evaluated under real-time constraints.

##### 2.2.1.4 Strengths

The study showed that real-time intention prediction is feasible with a relatively simple sensing setup. It demonstrated that intention inference can be useful even before an action is completed, and it highlighted latency as a critical design factor. The use of ROS integration also makes the work practically relevant for robotic systems.

##### 2.2.1.5 Limitations

The work focused on a specific reaching scenario rather than broader human movement classes. It also did not address explicit hand-command recognition, and its intention model was tied closely to the reaching task. As a result, the system was not designed for a broader multimodal intention-recognition framework involving both gestures and multiple body-movement classes.

##### 2.2.1.6 Relevance to the Present Study

This study is relevant because it demonstrates the practical value of early visual intention prediction and the importance of real-time integration. However, the present work extends beyond single-task reaching by combining gesture recognition with whole-body movement understanding and by incorporating stronger live-stability mechanisms for front/back motion interpretation.

#### 2.2.2 Review Two

##### 2.2.2.1 Research Topic and Author

3D Gesture Recognition and Adaptation for Human-Robot Interaction (Mahmud et al., 2022)

##### 2.2.2.2 Objectives

The objective of this study was to develop a three-dimensional gesture-recognition system for human-robot interaction capable of recognizing both pointing and dynamic gestures and using them for robot navigation-related interaction.

##### 2.2.2.3 Methodology

The system used Kinect skeletal tracking to capture joint coordinates from the human body and hand. Pointing direction was estimated geometrically, while dynamic gestures were recognized from temporal joint movement. Multiple machine-learning approaches were considered, and the recognized gestures were connected to robot-navigation actions in a simulation environment.

##### 2.2.2.4 Strengths

The study showed the value of vision-based gesture recognition in robotic interaction and demonstrated that directional and dynamic gestures can be interpreted in real time. It also highlighted the importance of structured spatial features and temporal information for gesture understanding.

##### 2.2.2.5 Limitations

The approach depended on Kinect hardware and was therefore constrained by that sensor's tracking behavior and deployment requirements. It was also centered primarily on gesture interpretation rather than a combined gesture-motion intention system. In addition, robot evaluation was limited to simulated behavior without addressing live body-motion interpretation in the same pipeline.

##### 2.2.2.6 Relevance to the Present Study

This work is relevant because it demonstrates the role of visual gesture recognition in HRI. The present study differs by using MediaPipe landmarks from a monocular camera rather than Kinect and by combining explicit gesture recognition with contextual body-motion analysis in the same runtime system.

#### 2.2.3 Review Three

##### 2.2.3.1 Research Topic and Author

Safe and Efficient Motion Planning for Material Transportation Robots Considering Intention Prediction of Obstacles (Li, Zhang et al., 2023)

##### 2.2.3.2 Objectives

The objective of the study was to improve the safety and efficiency of robot navigation in dynamic environments by predicting whether obstacles, including human workers, were likely to move away and using that information in navigation planning.

##### 2.2.3.3 Methodology

The system combined camera and LiDAR sensing to classify obstacles and estimate their likely movement behavior. Predicted obstacle intention was used to update the costmap used by the planner, allowing the robot to navigate more efficiently while maintaining safety. The system was evaluated in a simulation environment using ROS2-compatible infrastructure.

##### 2.2.3.4 Strengths

This study is particularly valuable because it treats obstacle behavior as an intention-related signal rather than assuming all obstacles are static or equally hazardous. It shows that movement direction and obstacle behavior can improve robot response quality without requiring a complete redesign of the navigation framework.

##### 2.2.3.5 Limitations

The work was focused more on navigation costmap adaptation than on multimodal intention recognition. It did not incorporate explicit gesture commands, and its perception design was not centered on a single-camera pose-recognition pipeline. The obstacle-intention model was also embedded in a specific navigation-planning context.

##### 2.2.3.6 Relevance to the Present Study

This work is highly relevant to the present study because it reinforces the importance of knowing whether a person is approaching or moving away from the robot. The present project adapts that idea at a perception level by using pose-derived depth and scale trends to improve front/back motion understanding before the robot action stage.

### 2.3 Summary of Literature Review and Identified Gap

The reviewed studies show that several important components of intention-aware robotics have already been explored: gesture recognition, early motion prediction, pose-based behavior estimation, and safer robot navigation in human environments. However, these works often isolate one capability at a time. Some concentrate on gestures, some focus on motion prediction, and others focus on navigation behavior around moving obstacles.

The main gap identified is the lack of a practical integrated system that combines explicit hand-command recognition with whole-body motion understanding in a live camera-based pipeline while also addressing the mismatch between offline model accuracy and real-time behavior. Many systems can classify movement or gestures under controlled conditions, but fewer studies demonstrate how to maintain stable front/back interpretation during live operation when pose noise and label noise affect the output.

This study addresses that gap by building a multimodal system that combines gesture recognition, LSTM-based movement classification, ST-GCN-based skeleton action modeling, and direction-aware front/back reasoning. The system was further strengthened by auditing and cleaning the forward/backward training data and by validating performance not only offline but also in live desktop and ROS2/Gazebo testing.

# CHAPTER THREE

## METHODOLOGY

### 3.0 Overview

This chapter presents the methodology used to design, implement, train, integrate, and evaluate the human intention recognition system developed in this study. The system was built as a vision-based pipeline that interprets hand gestures and body motion from a live camera stream and converts the recognized human state into robot response behavior. The methodology covers system architecture, data preparation, feature extraction, model development, runtime inference logic, ROS2 integration, and evaluation.

### 3.1 Research Design

The study adopted an implementation-based experimental design. Rather than testing a purely theoretical model, the project involved building a complete working system and evaluating it under live runtime conditions. The design process followed an iterative engineering workflow:

1. define the target human-intention classes and robot behaviors;
2. develop gesture and movement perception modules;
3. collect and organize training data;
4. train and evaluate machine-learning models;
5. analyze live runtime failures;
6. improve the inference pipeline based on observed weaknesses; and
7. validate the final system in desktop and ROS2/Gazebo modes.

This design was appropriate because the main challenge was not only classification accuracy but also practical reliability during live operation.

### 3.2 System Architecture

The final system consists of five main layers:

1. **Video Input Layer** - acquires live frames from a monocular camera.
2. **Landmark Extraction Layer** - uses MediaPipe Hands and MediaPipe Pose to extract hand and body landmarks.
3. **Recognition Layer** - performs:
   - hand gesture classification,
   - movement classification using an LSTM,
   - skeleton action classification using an ST-GCN.
4. **Inference Stabilization Layer** - applies temporal smoothing, confidence handling, stationary gating, and direction-aware front/back reasoning using signed pose depth and body-scale trends.
5. **Decision and Robot Response Layer** - combines gesture and movement results and generates robot action commands in desktop mode or ROS2 messages in robot mode.

The system treats gesture input as the highest-priority source because gestures represent explicit human commands. If no valid gesture is present, the system uses body movement and action recognition to infer contextual human state and select a robot response.

### 3.3 Data Sources and Dataset Preparation

The project used two main categories of data: gesture landmark samples and motion sequences.

#### 3.3.1 Gesture Data

Gesture recognition was based on hand-landmark data derived from MediaPipe. Each gesture sample consisted of hand landmark coordinates from a single detected hand, and advanced geometric features were extracted from these coordinates for classification. The final gesture classes used in the system were:

- `stop`
- `forward`
- `backward`
- `left`
- `right`

#### 3.3.2 Movement Data

Movement recognition used temporal body-pose sequences extracted from MediaPipe Pose. Each sequence represented a short window of body motion and was assigned to one of five classes:

- `approaching`
- `moving_away`
- `moving_left`
- `moving_right`
- `stationary`

The original recorded movement dataset contained label inconsistencies, especially in the forward/backward classes. To address this, the dataset was audited using direction cues derived from pose depth and body-scale behavior.

#### 3.3.3 Movement Direction Audit

An audit tool was developed to analyze whether recorded `approaching` and `moving_away` samples agreed with signed motion direction estimated from pose geometry. Each sample was categorized as:

- `agree`
- `ambiguous`
- `contradict`

The audit showed that the original dataset contained meaningful noise in the forward/backward labels. A cleaned dataset, `movement_clean`, was then created by retaining only agreeing samples for `approaching` and `moving_away`, while preserving all samples for `moving_left`, `moving_right`, and `stationary`.

This step improved the quality of training data used for the sequence models.

### 3.4 Feature Engineering

#### 3.4.1 Gesture Features

For gesture classification, the raw 3D hand landmarks produced by MediaPipe were transformed into advanced geometric features. These included normalized relative landmark relationships and shape-related descriptors that made the gesture classifier more robust than using only raw coordinates.

#### 3.4.2 Movement Features

For LSTM-based movement recognition, each pose sequence was converted into a temporal feature representation including:

- normalized landmark positions,
- joint velocity features for selected motion-relevant joints,
- approach/away features derived from torso and hip depth,
- body-width and body-scale cues over time.

These features were designed to preserve temporal information about motion direction and magnitude.

#### 3.4.3 Direction Features

To improve front/back motion recognition, a signed direction signature was computed from each pose sequence using:

- signed pose depth change, and
- signed body-scale change.

If the person became larger and closer in the camera view, this indicated likely `approaching` behavior. If the person became smaller and farther, this indicated likely `moving_away` behavior. These signals were used both during dataset auditing and during live inference.

### 3.5 Model Development

#### 3.5.1 Gesture Classifier

The gesture-recognition module uses a landmark-based machine-learning classifier trained on hand-landmark features. This classifier was chosen because gesture commands are relatively discrete and can be recognized effectively from structured hand geometry without requiring long temporal windows.

#### 3.5.2 Movement LSTM

The movement-recognition module uses a Long Short-Term Memory network trained on pose-derived temporal features. The LSTM was selected because it is effective for sequence classification where time order matters. The model was trained to classify the five movement classes listed above.

#### 3.5.3 ST-GCN Action Model

The system also includes a Spatial-Temporal Graph Convolutional Network trained directly on skeleton sequences. In this model, body joints are treated as graph nodes and their spatial and temporal relationships are learned jointly. The ST-GCN serves as a secondary action-recognition pathway that complements the LSTM.

### 3.6 Training Procedure

Model training was carried out using supervised learning. The movement LSTM and ST-GCN were trained on the cleaned movement dataset after the forward/backward audit step. Performance was monitored using standard classification metrics such as accuracy, precision, recall, and F1-score.

The movement LSTM produced strong but not perfectly reliable forward/backward results after retraining, while the ST-GCN provided a useful secondary motion signal. However, live testing showed that model output alone was still insufficient for the most reliable front/back interpretation. This motivated the final runtime design.

### 3.7 Live Inference and Runtime Stabilization

The runtime system processes a sliding window of pose sequences and applies temporal smoothing before generating a movement decision. However, several live issues were identified during testing:

- delayed switching between front/back classes,
- false stationary insertion during reversals,
- occasional contradiction between model output and obvious visual movement direction.

To address these issues, the runtime pipeline was extended with direction-aware stabilization logic:

1. signed depth and scale trends were estimated continuously from the pose sequence;
2. these signals were used to bias the front/back class probabilities;
3. when the direction signal was strong enough, it could directly determine the front/back decision between `approaching` and `moving_away`;
4. stationary predictions were suppressed when clear directional movement was still present.

This hybrid design was a major part of the final system because it improved behavior under real camera conditions without discarding the learned models.

### 3.8 Decision Logic and Robot Response

The robot decision layer uses a priority-based structure:

1. **Gesture Priority** - if a valid gesture is recognized with sufficient confidence, it overrides all motion-based behavior.
2. **Motion-Based Response** - if no gesture is active, the system uses movement and action recognition to determine the human state.
3. **Conservative Safety Policy** - the robot behaves cautiously in ambiguous or non-command situations.

The implemented robot behavior is as follows:

- `stop` gesture -> robot stops
- `forward` gesture -> robot moves forward
- `backward` gesture -> robot moves backward
- `left` gesture -> robot turns left
- `right` gesture -> robot turns right
- `approaching` -> robot stops and turns to avoid
- `moving_left` -> robot adjusts to the right
- `moving_right` -> robot adjusts to the left
- `moving_away` -> robot holds position
- `stationary` -> robot holds position

This policy was chosen to prioritize safety and predictable demonstration behavior.

### 3.9 ROS2 Integration

The final system was integrated into a ROS2 node for robotic deployment. The ROS2 node performs perception, inference, and decision-making, then publishes:

- `/cmd_vel`
- `/human_gesture`
- `/human_movement`
- `/human_action`

The ROS2 implementation shares the same direction-aware front/back logic as the desktop live system. It also supports a stable demo mode intended for cleaner runtime behavior during presentations and testing.

### 3.10 Evaluation Strategy

The system was evaluated in two complementary ways:

#### 3.10.1 Offline Evaluation

The trained gesture and movement models were evaluated using classification metrics such as accuracy, precision, recall, F1-score, and confusion matrices. This measured how well the models learned the intended classes from the prepared datasets.

#### 3.10.2 Live Evaluation

Because offline metrics did not fully capture runtime behavior, live evaluation was treated as a critical part of validation. Desktop live testing and screenshot-based inspection were used to verify:

- correct gesture override behavior,
- stable front/back switching,
- reduced false stationary transitions,
- consistency between current movement and displayed movement,
- acceptable robot response under camera input.

ROS2 and Gazebo testing were then used to verify that the same inference logic worked in a robot-control pipeline.

### 3.11 Summary

This methodology combined machine learning, feature engineering, dataset auditing, live runtime analysis, and ROS2 integration to build a practical intention-recognition system. The study did not rely solely on classifier accuracy; instead, it treated live reliability as a core engineering requirement. As a result, the final system emerged as a hybrid architecture in which learned models were strengthened by physically interpretable motion-direction reasoning, producing more reliable robot behavior in both desktop and ROS2-based operation.
