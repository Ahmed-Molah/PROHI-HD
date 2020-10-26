import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn import preprocessing

from flask import send_file
import io

import plotly.express as px

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
	'position': 'fixed',
	'top': 30,
	'left': 30,
	'bottom': 20,
	'width': '25%',
	'padding': '20px 10px',
	'background-color': '#f8f9fa',
	'overflow-x':'hidden',
	'overflow-y':'auto'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
	'margin-left': '28%',
	'margin-right': '5%',
	'padding': '20px 90p'
}

TEXT_STYLE = {
	'textAlign': 'center',
	'color': '#191970'
}

CARD_TEXT_STYLE = {
	'textAlign': 'center',
	'color': '#0074D9'
}

CARD_TEXT_STYLE2 = {
	'textAlign': 'left',
	'color': '#0074D9'
}

CARD_TEXT_STYLE3 = {
	'textAlign': 'center',
	'color': "light"
}

CARD_TEXT_STYLE4 = {
	'textAlign': 'center',
	'color': "warning"
}
padding_STYLE = {
  'padding-top': '12px',
  'padding-bottom': '18px'
}

upload_STYLE = {
  'width': '100%',
  'line-height': '30px',
  'border-width': '1px',
  'border-style': 'dashed',
  'border-radius': '5px',
  'text-align': 'center',
  'padding-top': '12px',
  'padding-bottom': '18px'
}

controls = dbc.FormGroup(
	[
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Patient Age", addon_type="prepend"),
				dbc.Input(id="styled-numeric-input_age", placeholder="Amount", type="number", min=18, max=120, step=1,
				value=['18'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Gender", addon_type="prepend"),
				dbc.Select(
					id='dropdown_sex',
					options=[
						{"label": "Male", "value": 1},
						{"label": "Female", "value": 0},
					],
					value=['1'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("chest pain type", addon_type="prepend"),
				dbc.Select(
				id='dropdown_cpt',
					options=[
						{"label": "typical angina", "value": 1},
						{"label": "atypical angina", "value": 2},
						{"label": "non-anginal pain", "value": 3},
						{"label": "asymptomatic", "value": 4},
					],
					value=['1'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Resting BP (mm Hg)", addon_type="prepend"),
				dbc.Input(id="styled-numeric-input_rbp", placeholder="Amount", type="number", min=80, max=200, step=1,
				value=['145']  # default value
				#dbc.InputGroupAddon("mm Hg", addon_type="append"
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Serum cholestoral in mg/dl", addon_type="prepend"),
				dbc.Input(id="styled-numeric-input_serum", placeholder="Amount", type="number", min=0, max=50, step=1,
				value=['10'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("fasting B sugar(>120 mg/d)", addon_type="prepend"),
				dbc.Select(
					id='dropdown_fbs',
					options=[
						{"label": "Yes", "value": 1},
						{"label": "No", "value": 0},
					],
					value=['1'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("RESTECG", addon_type="prepend"),
				dbc.Select(
				id='dropdown_restecg',
					options=[
						{"label": "normal", "value": 0},
						{"label": "having ST-T wave abnormality", "value": 1},
						{"label": "showing probable or definite left ventricular hypertrophy by Estes", "value": 2},
					],
					value=['0'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("max heart rate achieved", addon_type="prepend"),
				dbc.Input(id="styled-numeric-input_thalach", placeholder="Amount", type="number", min=60, max=190, step=1,
				value=['60'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Exercise induced angina", addon_type="prepend"),
				dbc.Select(
					id='dropdown_eia',
					options=[
						{"label": "Yes", "value": 1},
						{"label": "No", "value": 0},
					],
					value=['1'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("oldpeak (ST depression)", addon_type="prepend"),
				dbc.Input(id="styled-numeric-input_oldpeak", placeholder="Amount", type="number", min=2.6, max=5, step=0.1,
				value=['2.8'],  # default value

				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("NO of major vessels", addon_type="prepend"),
				dbc.Input(id="styled-numeric-input_ca", placeholder="Amount", type="number", min=0, max=3, step=1,
				value=['1'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Slope", addon_type="prepend"),
				dbc.Select(
				id='dropdown_Slope',
					options=[
						{"label": "upsloping", "value": 1},
						{"label": "flat", "value": 2},
						{"label": "downsloping", "value": 3},
					],
					value=['1'],  # default value
				),
			],
			className="mb-3",
		),
		dbc.InputGroup(
			[
				dbc.InputGroupAddon("Thal", addon_type="prepend"),
				dbc.Select(
				id='dropdown_thal',
					options=[
						{"label": "normal", "value": 3},
						{"label": "fixed defect", "value": 6},
						{"label": "reversable defect", "value": 7},
					],
					value=['3'],  # default value
				),
			],
			className="mb-3",
		),
		#html.Br(),
		dbc.Button(
			id='submit_button',
			n_clicks=0,
			children='Submit',
			color='primary',
			className="mr-1",
			size="lg",
			block=True
		),
		html.Br(),
		#html.Hr(),
		html.Div(
			children=[
				html.H6("Upload Patient data (CSV file)"),
				dcc.Upload(
				id="upload_patient_data",
				children=html.Div(
				children=[
					html.P("Drag and Drop or "),
					html.A("Select Files"),
							],
					style= upload_STYLE
					),
			accept=".csv",
			multiple=False,
				),
	    	#line-height = "60px"
				],
				),
		]
	)

html.Hr(),
sidebar = html.Div(
	[
		#html.Hr(),
		html.H2('Patient Data', style=TEXT_STYLE),
		html.Hr(),
		controls,
	],
	style=SIDEBAR_STYLE,
	)

content_first_row = dbc.Row([
	dbc.Col(
		dbc.Card(
			[
				dbc.CardBody(
					[
						html.H5("Clinical Analytics"),
						html.H3("Welcome to the Heart Disease Diagnosis Dashboard"),
						html.Div(
							id="intro",
							children="Predict clinic patient Heart Disease Presence and its intensity based on observed attributes of a patient and SVM Model. "
                                     "SVM Model was trained with UCI Heart Disease dataset available at the UCI Machine Learning data repository: http://archive.ics.uci.edu/ml/datasets/Heart+Disease."
                                     "Use Slidebar on the left to enter/upload patient data.",
						),
					]
				)
			],
		color="light"
		),
		#md=3,
		className="w-75 mb-3",
	)
])

content_second_row = dbc.Row(
	[
		dbc.Col(
			dbc.Card(
				[
					dbc.CardBody(
						[
							html.H4(id='card_patient_data', children=["Patient Data"], className='card-title', style=CARD_TEXT_STYLE),
							html.P(id='patient_data_age', children=[' Patient Age: '],
								   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_sex', children=[' Patient Gender: '],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_cpt', children=[' Chest pain type: '],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_rbp', children=[' Resting BP (mm Hg): '],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_serum', children=[' Serum cholestoral in mg/dl'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_fbs', children=[' Fasting B sugar(>120 mg/d)'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_restecg', children=[' RESTECG'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_thalach', children=[' Max heart rate achieved)'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_eia', children=[' Exercise induced angina'],
								   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_oldpeak', children=[' oldpeak (ST depression)'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_ca', children=[' NO of major vessels'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_Slope', children=[' Slope'],
															   style=CARD_TEXT_STYLE2),
							html.P(id='patient_data_thal', children=[' Thal'],
															   style=CARD_TEXT_STYLE2),
						]
					)
				]
			),
			#md=3,
			width=7
		),
		dbc.Col(
			dbc.Card(
				[
					dbc.CardBody(
						[
							html.H4(id='card_result', children=['Result'], className='card-title',
									style=CARD_TEXT_STYLE),
							html.P(id='card_calssification_result', children=[' No Heart Disease presence'],
								   style=CARD_TEXT_STYLE),
						]
					)
				],
		id= 'calssification_result',
		color="light",
		inverse=True,
		#outline=True
		#color="info",

			),
			className="mb-3",
			#md=3,
			width=5,
		)
	])

content = html.Div(
	[
		#html.H2('Heart Disease Diagnosis Dashboard', style=TEXT_STYLE),
		html.Hr(),
		content_first_row,
		html.Hr(),
		#html.Br(),
		#html.H2('Patient entered data', style=TEXT_STYLE),
		#html.Hr(),
		#html.Br(),
		content_second_row,
		html.Hr(),
		#html.Br(),
		html.Div(children=[
		html.A("Download patient data", href="/download_data/"),
		]),
	],
	style=CONTENT_STYLE
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = html.Div([sidebar, content])

@app.callback(
	Output('card_calssification_result', 'children'), Output('card_patient_data', 'children'), Output('calssification_result', 'color'),
	Output('patient_data_age', 'children'), Output('patient_data_sex', 'children'),Output('patient_data_cpt', 'children'),Output('patient_data_rbp', 'children'),
	Output('patient_data_serum', 'children'), Output('patient_data_fbs', 'children'),Output('patient_data_restecg', 'children'),Output('patient_data_thalach', 'children'),
	Output('patient_data_eia', 'children'),Output('patient_data_oldpeak', 'children'),Output('patient_data_ca', 'children'),Output('patient_data_Slope', 'children'),
	Output('patient_data_thal', 'children'),
	[Input('submit_button', 'n_clicks'), dash.dependencies.Input('upload_patient_data', 'contents'),
	dash.dependencies.Input('upload_patient_data', 'filename')],
	[State('styled-numeric-input_age', 'value'), State('dropdown_sex', 'value'), State('dropdown_cpt', 'value'),
	 State('styled-numeric-input_rbp', 'value'),State('styled-numeric-input_serum', 'value'),State('dropdown_fbs', 'value'),
	State('dropdown_restecg', 'value'),State('styled-numeric-input_thalach', 'value'),State('dropdown_eia', 'value'),
	State('styled-numeric-input_oldpeak', 'value'),State('styled-numeric-input_ca', 'value'),State('dropdown_Slope', 'value'),
	State('dropdown_thal', 'value'),
	 ])
def update_classification_result(n_clicks, upload_patient_contents, upload_patient_filename, age_value, sex_value, cpt_value, rbp_value, serum_value, fbs_value, restecg_value, thalach_value, eia_value, oldpeak_value, ca_value,
				   Slope_value, thal_value):
	calssification_result = "No Heart Disease presence"
	CARD_STYLE = "light"

	user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

	if user_click == 'submit_button':  # upload_patient_data
		# do something
		print('user_click value ............')
		print(user_click)

		if n_clicks is None:
			raise PreventUpdate
		else:
			if n_clicks > 0:
				print(n_clicks)
				print(age_value)
				print(sex_value)
				print(cpt_value)
				print(rbp_value)
				print(serum_value)
				print(fbs_value)
				print(restecg_value)
				print(thalach_value)
				print(eia_value)
				print(oldpeak_value)
				print(ca_value)
				print(Slope_value)
				print(thal_value)

				gender = 'Female' if sex_value == '0' else 'Male'
				restecg = 'normal' if restecg_value == '0' else 'having ST-T wave abnormality' if restecg_value == '1' else 'showing probable or definite left ventricular hypertrophy by Estes'
				cp = 'typical angina' if cpt_value == '1' else 'atypical angina' if cpt_value == '2' else 'non-anginal pain' if cpt_value == '3' else 'asymptomatic'
				fbs = 'Yes' if fbs_value == '1' else 'No'
				eia = 'Yes' if eia_value == '1' else 'No'
				slope = 'upsloping' if Slope_value == '1' else 'flat' if Slope_value == '2' else 'downsloping'
				thal = 'normal' if thal_value == '3' else 'fixed defect' if thal_value == '6' else 'reversable defect'

				print(age_value)

				str1 = " "
				#age_value_new = str1.join(age_value)
				#rbp_value_new = str1.join(rbp_value)
				#serum_value_new = str1.join(serum_value)
				#thalach_value_new = str1.join(thalach_value)
				#oldpeak_value_new = str1.join(oldpeak_value)
				#ca_value_new = str1.join(ca_value)

				#print(age_value_new)

				patient_entered_data_output = 'Patient Age: ' + str(age_value), 'Patient Gender: ' + gender, 'Chest pain type: ' + cp, \
											  'Resting BP (mm Hg): ' + str(rbp_value) , 'Serum cholestoral in mg/dl: ' + str(serum_value), \
											  'Fasting B sugar(>120 mg/d): ' + str(fbs), 'RESTECG: ' + str(restecg), 'Max heart rate achieved): ' + str(thalach_value),\
											  'Exercise induced angina: ' + str(eia), 'oldpeak (ST depression): ' + str(oldpeak_value), 'NO of major vessels: ' + str(ca_value), 'Slope: ' + str(slope) ,\
											  'Thal: ' + str(thal)

				# Loading Dataset
				# patients_data = pd.read_csv("processed.cleveland.data")
				patients_data = pd.read_csv("processed.cleveland.data")

				X, Y = data_preprocessing(patients_data)

				# Create a svm Classifier
				SVM_classfier = svm.SVC(kernel='linear')  # Linear Kernel only

				# Train the model
				SVM_classfier.fit(X, Y)

				# report_df = pd.read_csv(OUTPUT_DATA_FILEPATH + 'report_df.csv').set_index('classifier')

				if sex_value == 'Male':
					sex_value = '0'
				else:
					sex_value = '1'

				target_value = '0'
				# Initialise patient data as a list.
				# current_patient_data = [{'age': age_value, 'sex': sex_value, 'cp': cpt_value, 'trestbps': rbp_value, 'chol': serum_value, 'fbs': fbs_value, 'restecg': restecg_value, 'thalach': thalach_value, 'exang': eia_value, 'oldpeak': oldpeak_value, 'slope': Slope_value, 'ca': ca_value, 'thal': thal_value, 'goal': target_value }]

				# Creates DataFrame.
				current_patient_data_df = pd.DataFrame(
					{'age': age_value, 'sex': sex_value, 'cp': cpt_value, 'trestbps': rbp_value, 'chol': serum_value,
					 'fbs': fbs_value, 'restecg': restecg_value, 'thalach': thalach_value, 'exang': eia_value,
					 'oldpeak': oldpeak_value, 'slope': Slope_value, 'ca': ca_value, 'thal': thal_value,
					 'goal': target_value}, index=[0])

				print('Current patient data values....')
				print(current_patient_data_df)

				data_features, data_target = data_preprocessing(current_patient_data_df)

				# Predict the response for test dataset
				y_pred = SVM_classfier.predict(data_features)

				# Print the data
				print('Classification value')
				print(y_pred)

				if y_pred == 0:
					calssification_result = 'No Heart Disease presence'
					# CARD_STYLE = "success"
					CARD_STYLE = "primary"
					CARD_STYLE = "light"
				else:
					CARD_STYLE = "warning"
					calssification_result = 'Heart Disease presence'

			return_value = calssification_result, 'Current Patient Data', CARD_STYLE, patient_entered_data_output

			print('Final patient classification....')
			print(calssification_result)

			return calssification_result, 'Current Patient Data', CARD_STYLE, 'Patient Age: ' + str(age_value), 'Patient Gender: ' + gender, 'Chest pain type: ' + cp, \
											  'Resting BP (mm Hg): ' + str(rbp_value) , 'Serum cholestoral in mg/dl: ' + str(serum_value), \
											  'Fasting B sugar(>120 mg/d): ' + str(fbs), 'RESTECG: ' + str(restecg), 'Max heart rate achieved): ' + str(thalach_value),\
											  'Exercise induced angina: ' + str(eia), 'oldpeak (ST depression): ' + str(oldpeak_value), 'NO of major vessels: ' + str(ca_value), 'Slope: ' + str(slope) ,\
											  'Thal: ' + str(thal)
	else:
		print(user_click)
		if upload_patient_contents is not None:
			upload_patient_df = pd.read_csv(upload_patient_filename)
			df_list = upload_patient_df.loc[0, :]

			print(df_list)
			age_upload_value = df_list['age']
			sex_upload_value = df_list['sex']
			cp_upload_value = df_list['cp']
			rbp_upload_value = df_list['trestbps']
			chol_upload_value = df_list['chol']
			fbs_upload_value = df_list['fbs']
			restecg_upload_value = df_list['restecg']
			thalach_upload_value = df_list['thalach']
			eia_upload_value = df_list['exang']
			oldpeak_upload_value = df_list['oldpeak']
			slope_upload_value = df_list['slope']
			ca_upload_value = df_list['ca']
			thal_upload_value = df_list['thal']
			goal_upload_value = df_list['goal']

			print(df_list['age'])
			print(df_list['sex'])

			gender = 'Female' if sex_upload_value == '0' else 'Male'
			restecg = 'normal' if restecg_upload_value == '0' else 'having ST-T wave abnormality' if restecg_upload_value == '1' else 'showing probable or definite left ventricular hypertrophy by Estes'
			cp = 'typical angina' if cp_upload_value == '1' else 'atypical angina' if cp_upload_value == '2' else 'non-anginal pain' if cp_upload_value == '3' else 'asymptomatic'
			fbs = 'Yes' if fbs_upload_value == '1' else 'No'
			eia = 'Yes' if eia_upload_value == '1' else 'No'
			slope = 'upsloping' if slope_upload_value == '1' else 'flat' if slope_upload_value == '2' else 'downsloping'
			thal = 'normal' if thal_upload_value == '3' else 'fixed defect' if thal_upload_value == '6' else 'reversable defect'

			patient_uploaded_data_output = 'Patient Age: ' + str(age_upload_value), 'Patient Gender: ' + str(
				gender), 'Chest pain type: ' + str(cp), 'Resting BP (mm Hg): ' + str(
				age_upload_value), 'Serum cholestoral in mg/dl: ' + str(age_upload_value), 'Fasting B sugar(>120 mg/d): ' + str(
				fbs), 'RESTECG: ' + str(restecg), 'Max heart rate achieved): ' + str(
				age_upload_value), 'Exercise induced angina: ' + str(eia), 'oldpeak (ST depression): ' + str(
				age_upload_value), 'NO of major vessels: ' + str(age_upload_value), 'Slope: ' + str(slope), 'Thal: ' + str(thal)

			#return patient_data_output

			# Loading Dataset
			# patients_data = pd.read_csv("processed.cleveland.data")
			patients_data = pd.read_csv("processed.cleveland.data")

			X, Y = data_preprocessing(patients_data)

			# Create a svm Classifier
			SVM_classfier = svm.SVC(kernel='linear')  # Linear Kernel only

			# Train the model
			SVM_classfier.fit(X, Y)

			# report_df = pd.read_csv(OUTPUT_DATA_FILEPATH + 'report_df.csv').set_index('classifier')

			if sex_value == 'Male':
				sex_value = '0'
			else:
				sex_value = '1'

			target_value = '0'
			# Initialise patient data as a list.

			print('Current Uploaded patient data values....')
			print(upload_patient_df)

			data_features, data_target = data_preprocessing(upload_patient_df)

			# Predict the response for test dataset
			y_pred = SVM_classfier.predict(data_features)

			# Print the data
			print('Classification value')
			print(y_pred)

			if y_pred == 0:
				calssification_result = "No Heart Disease presence"
				# CARD_STYLE = "success"
				CARD_STYLE = "primary"
				CARD_STYLE = "light"
			else:
				CARD_STYLE = "warning"
				calssification_result = "Heart Disease presence"

			return_value = calssification_result, 'Current uploaded Patient Data', CARD_STYLE, 'Patient Age: ' + str(int(age_upload_value)), 'Patient Gender: ' + str(
				gender), 'Chest pain type: ' + str(cp), 'Resting BP (mm Hg): ' + str(
				age_upload_value), 'Serum cholestoral in mg/dl: ' + str(age_upload_value), 'Fasting B sugar(>120 mg/d): ' + str(
				fbs), 'RESTECG: ' + str(restecg), 'Max heart rate achieved): ' + str(
				age_upload_value), 'Exercise induced angina: ' + str(eia), 'oldpeak (ST depression): ' + str(
				age_upload_value), 'NO of major vessels: ' + str(age_upload_value), 'Slope: ' + str(slope), 'Thal: ' + str(thal)

			print('Final patient classification....')
			print(calssification_result)
			return return_value

		gender = 'Female' if sex_value == '0' else 'Male'
		restecg = 'normal' if restecg_value == '0' else 'having ST-T wave abnormality' if restecg_value == '1' else 'showing probable or definite left ventricular hypertrophy by Estes'
		cp = 'typical angina' if cpt_value == '1' else 'atypical angina' if cpt_value == '2' else 'non-anginal pain' if cpt_value == '3' else 'asymptomatic'
		fbs = 'Yes' if fbs_value == '1' else 'No'
		eia = 'Yes' if eia_value == '1' else 'No'
		slope = 'upsloping' if Slope_value == '1' else 'flat' if Slope_value == '2' else 'downsloping'
		thal = 'normal' if thal_value == '3' else 'fixed defect' if thal_value == '6' else 'reversable defect'

		# initialize an empty string
		str1 = " "
		age_value_new = str1.join(age_value)
		rbp_value_new = str1.join(rbp_value)
		serum_value_new = str1.join(serum_value)
		thalach_value_new = str1.join(thalach_value)
		oldpeak_value_new = str1.join(oldpeak_value)
		ca_value_new = str1.join(ca_value)

		print(age_value_new)

		patient_default_data_output = 'Patient Age: ' + age_value_new, 'Patient Gender: ' + str(
			gender), 'Chest pain type: ' + str(cp), 'Resting BP (mm Hg): ' + str(
			rbp_value_new), 'Serum cholestoral in mg/dl: ' + str(serum_value_new), 'Fasting B sugar(>120 mg/d): ' + str(
			fbs), 'RESTECG: ' + str(restecg), 'Max heart rate achieved): ' + str(
			thalach_value_new), 'Exercise induced angina: ' + str(eia), 'oldpeak (ST depression): ' + str(
			oldpeak_value_new), 'NO of major vessels: ' + str(ca_value_new), 'Slope: ' + str(slope), 'Thal: ' + str(
			thal)

		return_value = calssification_result, 'Current Patient Data', CARD_STYLE, 'Patient Age: ' + age_value_new, 'Patient Gender: ' + str(
			gender), 'Chest pain type: ' + str(cp), 'Resting BP (mm Hg): ' + str(
			rbp_value_new), 'Serum cholestoral in mg/dl: ' + str(serum_value_new), 'Fasting B sugar(>120 mg/d): ' + str(
			fbs), 'RESTECG: ' + str(restecg), 'Max heart rate achieved): ' + str(
			thalach_value_new), 'Exercise induced angina: ' + str(eia), 'oldpeak (ST depression): ' + str(
			oldpeak_value_new), 'NO of major vessels: ' + str(ca_value_new), 'Slope: ' + str(slope), 'Thal: ' + str(
			thal)
		return return_value

def data_preprocessing(patients_data):
	# Data cleaning and preprocessing
	patients = patients_data.copy()

	patients = patients.replace('?', np.nan)
	patients.ca = patients.ca.apply(pd.to_numeric)
	patients.thal = patients.thal.apply(pd.to_numeric)

	patients_data = patients.copy()
	patients_data.fillna(patients_data.median(), inplace=True)

	# Data Normailization
	x = patients_data.values

	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	patients_data_norm = pd.DataFrame(x_scaled, columns=patients_data.columns)

	patients_data_norm['sex'] = patients_data.sex.astype(int)
	patients_data_norm['fbs'] = patients_data.sex.astype(int)
	patients_data_norm['exang'] = patients_data.sex.astype(int)
	patients_data_norm['target'] = patients_data.sex.astype(int)

	# splitting data for training and testing

	patients = patients_data.copy()
	X = patients.drop('goal', axis=1)
	Y = patients['goal']

	return X, Y

def classification_data_mapping(model_calssification_result):

	if model_calssification_result == 1:
		final_result = 'HD Intensity 1'
	elif model_calssification_result == 2:
		final_result = 'HD Intensity 2'
	elif model_calssification_result == 3:
		final_result = 'HD Intensity 3'
	else:
		final_result = 'HD Intensity 4'

	return final_result

#@app.callback(
#	Output('card_patient_data', 'children'),
    #   [dash.dependencies.Input('upload_patient_data', 'contents'),
	#dash.dependencies.Input('upload_patient_data', 'filename')])

#def update_card_title_1(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    #	print('Uploaded Patient Data')
	#return 'Uploaded Patient Data'

@app.server.route('/download_data/')
def download_data():
	#Create DF
	d = {'Age': 20, 'Sex': 1}
	df = pd.DataFrame(data=d)

	#Convert DF
	strIO = io.BytesIO()
	excel_writer = pd.ExcelWriter(strIO, engine="xlsxwriter")
	df.to_excel(excel_writer, sheet_name="sheet1")
	excel_writer.save()
	excel_data = strIO.getvalue()
	strIO.seek(0)

	return send_file(strIO,
					 attachment_filename='patient-data.xlsx',
					 as_attachment=True)

if __name__ == '__main__':
	#app.run_server(debug=True)
    app.server.run(debug=True)