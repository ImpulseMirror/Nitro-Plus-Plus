# Nitro Plus Plus
Technology has caught up with the lazy. This is an experiment in what is possible when you choose not the "support the official translation"


## Poetry Commands
## Install dependencies and build wheel
```
poetry lock
poetry install -with cuda
```
### Start request server
```
poetry run nitropp serve --model org/repo
```
### Kill server 

```
poetry run killport 8787 # kills the process on port 8787
```

# Translation Endpoints

## Example Request

```
curl --location 'http://127.0.0.1:8787/translate' \
--header 'Content-Type: application/json; charset=utf-8' \
--data '{"text":"双子の兄達は3歳で魔力が発現しました。"}'
```
## Response

```
{
    "source": "双子の兄達は3歳で魔力が発現しました。",
    "translation": "The twin boys both developed magic powers at the age of three."
}
```

