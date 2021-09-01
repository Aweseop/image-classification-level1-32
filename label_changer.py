from enum import Enum

class IncorrectLabel(int, Enum):
	F2M = 0
	M2F = 1
	INCORRECT2NORMAL = 2

incorrect = {
	'006359': IncorrectLabel.F2M, 
	'006360': IncorrectLabel.F2M, 
	'006361': IncorrectLabel.F2M,
	'006362': IncorrectLabel.F2M,
	'006363': IncorrectLabel.F2M, 
	'006364': IncorrectLabel.F2M,
	'001498-1': IncorrectLabel.M2F,
	'004432' : IncorrectLabel.M2F,
	'000020' : IncorrectLabel.INCORRECT2NORMAL,
	'004418' : IncorrectLabel.INCORRECT2NORMAL,
	'005227' : IncorrectLabel.INCORRECT2NORMAL
}

class StateChanger:
	def __init__(self, state:IncorrectLabel):
		self.state = state
	
	def __call__(self, file_name, gender, age):
		if self.state == IncorrectLabel.F2M:
			return self.stateF2M(file_name, gender, age)
		if self.state == IncorrectLabel.M2F:
			return self.stateM2F(file_name, gender, age)
		if self.state == IncorrectLabel.INCORRECT2NORMAL:
			return self.stateI2N(file_name, gender, age)
		
		raise "여긴 또 왜왔냐"

	def stateF2M(self, file_name, gender, age):
		return (file_name, "male", age)

	def stateM2F(self, file_name, gender, age):
		return (file_name, "female", age)

	def stateI2N(self, file_name, gender, age):
		realName = file_name
		if file_name == "incorrect_mask":
			realName = "normal"
		if file_name == "normal":
			realName = "incorrect_mask"

		return (realName, gender, age)


def labelChanger(profile, _file_name, gender, age, _file_names):
	mask_label = _file_names[_file_name]
	manNum = profile.split('_')[0]

	if manNum in incorrect.keys():
		changer = StateChanger(incorrect[manNum])
		(_file_name, gender, age) = changer(_file_name, gender, age)
		mask_label = _file_names[_file_name]
	
	return (mask_label, gender, age)

