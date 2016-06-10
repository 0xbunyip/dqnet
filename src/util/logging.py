import os

def get_next_name(dir, name, ext):
	i = 0
	while True:
		new_name = name
		if i > 0:
			new_name += '_' + str(i)

		if len(ext) > 0:
			new_name += '.' + ext
		
		try:
			os.stat(os.path.join(dir, new_name))
		except OSError:
			return new_name
		i += 1

def write_info(f, c):
	hyper_params = [attr for attr in dir(c) \
		if not attr.startswith("__") and not callable(getattr(c, attr))]
	for param in hyper_params:
		f.write(str(c.__name__) + '.' + param + ' = ' + \
				str(getattr(c, param)) + '\n')
	f.write('\n')
