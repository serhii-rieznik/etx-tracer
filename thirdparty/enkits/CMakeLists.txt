cmake_minimum_required(VERSION 3.5)

project( enkiTS )

set( ENKITS_TASK_PRIORITIES_NUM "3" CACHE STRING "Number of task priorities, 1-5, 0 for defined by defaults in source" ) 

set( ENKITS_SRC
     LockLessMultiReadPipe.hxx
     TaskScheduler.hxx
     TaskScheduler.cxx
   )

list( APPEND ENKITS_SRC ${ENKITS_HEADERS} )

add_library( enkiTS STATIC ${ENKITS_SRC} )
target_include_directories( enkiTS PUBLIC "${CMAKE_CURRENT_LIST_DIR}" )
set_target_properties(enkiTS PROPERTIES FOLDER "thirdparty")

if( ENKITS_TASK_PRIORITIES_NUM GREATER "0" )
    target_compile_definitions( enkiTS PUBLIC "ENKITS_TASK_PRIORITIES_NUM=${ENKITS_TASK_PRIORITIES_NUM}" )    
endif()

if( UNIX )
    set( CMAKE_THREAD_PREFER_PTHREAD TRUE )
    find_package( Threads REQUIRED )
    if( CMAKE_USE_PTHREADS_INIT )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread" )
    endif()
    target_link_libraries( enkiTS ${CMAKE_THREAD_LIBS_INIT} )

    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif()

if( APPLE )
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" )
endif()
