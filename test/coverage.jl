using Coverage

coverage_data = process_folder("src")
LCOV.writefile("coverage/lcov.info", coverage_data)
