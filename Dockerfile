FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY diamonds.py ${LAMBDA_TASK_ROOT}
COPY retrievers.py ${LAMBDA_TASK_ROOT}
COPY tools.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY classes.py ${LAMBDA_TASK_ROOT}
COPY agents.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
RUN pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "diamonds.lambda_handler" ]