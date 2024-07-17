FROM julia:1.10
WORKDIR /app
COPY . .
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'
CMD ["julia", "--project=."]