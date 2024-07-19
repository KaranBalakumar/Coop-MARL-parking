cmake_minimum_required(VERSION 3.5)
project(robot_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    scripts
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
  scripts/marl_vdn_train.py
  scripts/marl_vdn_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
  scripts/configs/arguments.py
  scripts/configs/vdn_config.py
  DESTINATION lib/${PROJECT_NAME}/configs/
)

install(
  PROGRAMS
  scripts/marltoolkit/agents/base_agent.py
  scripts/marltoolkit/agents/__init__.py
  scripts/marltoolkit/agents/vdn_agent.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/agents/
)

install(
  PROGRAMS
  scripts/marltoolkit/data/__init__.py
  scripts/marltoolkit/data/ma_replaybuffer.py
  scripts/marltoolkit/data/replay_buffer.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/data/
)

install(
  PROGRAMS
  scripts/marltoolkit/modules/__init__.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/modules/
)

install(
  PROGRAMS
  scripts/marltoolkit/modules/actors/__init__.py
  scripts/marltoolkit/modules/actors/mlp.py
  scripts/marltoolkit/modules/actors/rnn.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/modules/actors/
)

install(
  PROGRAMS
  scripts/marltoolkit/modules/mixers/__init__.py
  scripts/marltoolkit/modules/mixers/vdn.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/modules/mixers/
)

install(
  PROGRAMS
  scripts/marltoolkit/modules/__init__.py
  scripts/marltoolkit/modules/actors/__init__.py
  scripts/marltoolkit/modules/actors/mlp.py
  scripts/marltoolkit/modules/actors/rnn.py
  scripts/marltoolkit/modules/mixers/__init__.py
  scripts/marltoolkit/modules/mixers/vdn.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/modules/
)

install(
  PROGRAMS
  scripts/marltoolkit/runners/episode_runner.py
  scripts/marltoolkit/runners/__init__.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/runners/
)

install(
  PROGRAMS
  scripts/marltoolkit/utils/__init__.py
  scripts/marltoolkit/utils/lr_scheduler.py
  scripts/marltoolkit/utils/model_utils.py
  scripts/marltoolkit/utils/progressbar.py
  scripts/marltoolkit/utils/timer.py
  scripts/marltoolkit/utils/transforms.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/utils/
)

install(
  PROGRAMS
  scripts/marltoolkit/utils/logger/__init__.py
  scripts/marltoolkit/utils/logger/base.py
  scripts/marltoolkit/utils/logger/base_orig.py
  scripts/marltoolkit/utils/logger/util_logging.py
  scripts/marltoolkit/utils/logger/logs.py
  scripts/marltoolkit/utils/logger/tensorboard.py
  DESTINATION lib/${PROJECT_NAME}/marltoolkit/utils/logger/
)

ament_package()
