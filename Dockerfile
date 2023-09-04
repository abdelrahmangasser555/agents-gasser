FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY biznes-clinics.py ${LAMBDA_TASK_ROOT}
COPY clinics.py ${LAMBDA_TASK_ROOT}
COPY retrievers.py ${LAMBDA_TASK_ROOT}
COPY tools.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY classes.py ${LAMBDA_TASK_ROOT}
COPY agents.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
RUN pip install -r requirements.txt

# add OPENAI_API_KEY environment variable and value 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
ENV OPENAI_API_KEY=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
ENV PINECONE_API_KEY=c869cafc-6f9a-4abf-b8ca-d24ebc2f6ccd
# aws credentials
ENV AWS_ACCESS_KEY_ID=AKIARCJAMQQJ5G6QLONK
# aws secret access key
ENV AWS_SECRET_ACCESS_KEY=XHOw10uOTNs033pQN7riZlDiAJpq9Je51yW7JoeU
ENV AWS_DEFAULT_REGION=us-west-1

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "biznes-clinics.lambda_handler" ]