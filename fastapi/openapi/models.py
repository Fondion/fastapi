from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union, NamedTuple, TypedDict

from fastapi._compat import (
    AnyUrl,
    AnyUrl_V1,
    BaseModel,
    BaseModel_V1,
    Field,
    Field_V1,
    EmailStr,
    EmailStr_V1,
    PYDANTIC_V2,
    CoreSchema,
    GetJsonSchemaHandler,
    JsonSchemaValue,
    _model_rebuild,
    with_info_plain_validator_function,
)
from fastapi.logger import logger
# from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from typing_extensions import deprecated as typing_deprecated


class BaseModelWithConfig(BaseModel):
    model_config = {"extra": "allow"}


class BaseModelWithConfig_V1(BaseModel_V1):
    class Config:
        extra = "allow"


class Contact(BaseModelWithConfig):
    name: Optional[str] = None
    url: Optional[AnyUrl] = None
    email: Optional[EmailStr] = None


class Contact_V1(BaseModelWithConfig_V1):
    name: Optional[str] = None
    url: Optional[AnyUrl_V1] = None
    email: Optional[EmailStr_V1] = None


class License(BaseModelWithConfig):
    name: str
    identifier: Optional[str] = None
    url: Optional[AnyUrl] = None


class License_V1(BaseModelWithConfig_V1):
    name: str
    identifier: Optional[str] = None
    url: Optional[AnyUrl_V1] = None


class Info(BaseModelWithConfig):
    title: str
    summary: Optional[str] = None
    description: Optional[str] = None
    termsOfService: Optional[str] = None
    contact: Optional[Contact] = None
    license: Optional[License] = None
    version: str


class Info_V1(BaseModelWithConfig_V1):
    title: str
    summary: Optional[str] = None
    description: Optional[str] = None
    termsOfService: Optional[str] = None
    contact: Optional[Contact] = None
    license: Optional[License] = None
    version: str


class ServerVariable(BaseModelWithConfig):
    enum: Annotated[Optional[List[str]], Field(min_length=1)] = None
    default: str
    description: Optional[str] = None


class ServerVariable_V1(BaseModelWithConfig_V1):
    enum: Annotated[Optional[List[str]], Field_V1(min_length=1)] = None
    default: str
    description: Optional[str] = None


class Server(BaseModelWithConfig):
    url: Union[AnyUrl, str]
    description: Optional[str] = None
    variables: Optional[Dict[str, ServerVariable]] = None


class Server_V1(BaseModelWithConfig_V1):
    url: Union[AnyUrl_V1, str]
    description: Optional[str] = None
    variables: Optional[Dict[str, ServerVariable]] = None


class Reference(BaseModel):
    ref: str = Field(alias="$ref")


class Reference_V1(BaseModel_V1):
    ref: str = Field_V1(alias="$ref")


class Discriminator(BaseModel):
    propertyName: str
    mapping: Optional[Dict[str, str]] = None


class Discriminator_V1(BaseModel_V1):
    propertyName: str
    mapping: Optional[Dict[str, str]] = None


class XML(BaseModelWithConfig):
    name: Optional[str] = None
    namespace: Optional[str] = None
    prefix: Optional[str] = None
    attribute: Optional[bool] = None
    wrapped: Optional[bool] = None


class XML_V1(BaseModelWithConfig_V1):
    name: Optional[str] = None
    namespace: Optional[str] = None
    prefix: Optional[str] = None
    attribute: Optional[bool] = None
    wrapped: Optional[bool] = None


class ExternalDocumentation(BaseModelWithConfig):
    description: Optional[str] = None
    url: AnyUrl


class ExternalDocumentation_V1(BaseModelWithConfig_V1):
    description: Optional[str] = None
    url: AnyUrl_V1


class Schema(BaseModelWithConfig):
    # Ref: JSON Schema 2020-12: https://json-schema.org/draft/2020-12/json-schema-core.html#name-the-json-schema-core-vocabu
    # Core Vocabulary
    schema_: Optional[str] = Field(default=None, alias="$schema")
    vocabulary: Optional[str] = Field(default=None, alias="$vocabulary")
    id: Optional[str] = Field(default=None, alias="$id")
    anchor: Optional[str] = Field(default=None, alias="$anchor")
    dynamicAnchor: Optional[str] = Field(default=None, alias="$dynamicAnchor")
    ref: Optional[str] = Field(default=None, alias="$ref")
    dynamicRef: Optional[str] = Field(default=None, alias="$dynamicRef")
    defs: Optional[Dict[str, "SchemaOrBool"]] = Field(default=None, alias="$defs")
    comment: Optional[str] = Field(default=None, alias="$comment")
    # Ref: JSON Schema 2020-12: https://json-schema.org/draft/2020-12/json-schema-core.html#name-a-vocabulary-for-applying-s
    # A Vocabulary for Applying Subschemas
    allOf: Optional[List["SchemaOrBool"]] = None
    anyOf: Optional[List["SchemaOrBool"]] = None
    oneOf: Optional[List["SchemaOrBool"]] = None
    not_: Optional["SchemaOrBool"] = Field(default=None, alias="not")
    if_: Optional["SchemaOrBool"] = Field(default=None, alias="if")
    then: Optional["SchemaOrBool"] = None
    else_: Optional["SchemaOrBool"] = Field(default=None, alias="else")
    dependentSchemas: Optional[Dict[str, "SchemaOrBool"]] = None
    prefixItems: Optional[List["SchemaOrBool"]] = None
    # TODO: uncomment and remove below when deprecating Pydantic v1
    # It generales a list of schemas for tuples, before prefixItems was available
    # items: Optional["SchemaOrBool"] = None
    items: Optional[Union["SchemaOrBool", List["SchemaOrBool"]]] = None
    contains: Optional["SchemaOrBool"] = None
    properties: Optional[Dict[str, "SchemaOrBool"]] = None
    patternProperties: Optional[Dict[str, "SchemaOrBool"]] = None
    additionalProperties: Optional["SchemaOrBool"] = None
    propertyNames: Optional["SchemaOrBool"] = None
    unevaluatedItems: Optional["SchemaOrBool"] = None
    unevaluatedProperties: Optional["SchemaOrBool"] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-structural
    # A Vocabulary for Structural Validation
    type: Optional[str] = None
    enum: Optional[List[Any]] = None
    const: Optional[Any] = None
    multipleOf: Optional[float] = Field(default=None, gt=0)
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    maxLength: Optional[int] = Field(default=None, ge=0)
    minLength: Optional[int] = Field(default=None, ge=0)
    pattern: Optional[str] = None
    maxItems: Optional[int] = Field(default=None, ge=0)
    minItems: Optional[int] = Field(default=None, ge=0)
    uniqueItems: Optional[bool] = None
    maxContains: Optional[int] = Field(default=None, ge=0)
    minContains: Optional[int] = Field(default=None, ge=0)
    maxProperties: Optional[int] = Field(default=None, ge=0)
    minProperties: Optional[int] = Field(default=None, ge=0)
    required: Optional[List[str]] = None
    dependentRequired: Optional[Dict[str, Set[str]]] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-vocabularies-for-semantic-c
    # Vocabularies for Semantic Content With "format"
    format: Optional[str] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-the-conten
    # A Vocabulary for the Contents of String-Encoded Data
    contentEncoding: Optional[str] = None
    contentMediaType: Optional[str] = None
    contentSchema: Optional["SchemaOrBool"] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-basic-meta
    # A Vocabulary for Basic Meta-Data Annotations
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    deprecated: Optional[bool] = None
    readOnly: Optional[bool] = None
    writeOnly: Optional[bool] = None
    examples: Optional[List[Any]] = None
    # Ref: OpenAPI 3.1.0: https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#schema-object
    # Schema Object
    discriminator: Optional[Discriminator] = None
    xml: Optional[XML] = None
    externalDocs: Optional[ExternalDocumentation] = None
    example: Annotated[
        Optional[Any],
        typing_deprecated(
            "Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, "
            "although still supported. Use examples instead."
        ),
    ] = None


class Schema_V1(BaseModelWithConfig_V1):
    # Ref: JSON Schema 2020-12: https://json-schema.org/draft/2020-12/json-schema-core.html#name-the-json-schema-core-vocabu
    # Core Vocabulary
    schema_: Optional[str] = Field_V1(default=None, alias="$schema")
    vocabulary: Optional[str] = Field_V1(default=None, alias="$vocabulary")
    id: Optional[str] = Field_V1(default=None, alias="$id")
    anchor: Optional[str] = Field_V1(default=None, alias="$anchor")
    dynamicAnchor: Optional[str] = Field_V1(default=None, alias="$dynamicAnchor")
    ref: Optional[str] = Field_V1(default=None, alias="$ref")
    dynamicRef: Optional[str] = Field_V1(default=None, alias="$dynamicRef")
    defs: Optional[Dict[str, "SchemaOrBool"]] = Field_V1(default=None, alias="$defs")
    comment: Optional[str] = Field_V1(default=None, alias="$comment")
    # Ref: JSON Schema 2020-12: https://json-schema.org/draft/2020-12/json-schema-core.html#name-a-vocabulary-for-applying-s
    # A Vocabulary for Applying Subschemas
    allOf: Optional[List["SchemaOrBool"]] = None
    anyOf: Optional[List["SchemaOrBool"]] = None
    oneOf: Optional[List["SchemaOrBool"]] = None
    not_: Optional["SchemaOrBool"] = Field_V1(default=None, alias="not")
    if_: Optional["SchemaOrBool"] = Field_V1(default=None, alias="if")
    then: Optional["SchemaOrBool"] = None
    else_: Optional["SchemaOrBool"] = Field_V1(default=None, alias="else")
    dependentSchemas: Optional[Dict[str, "SchemaOrBool"]] = None
    prefixItems: Optional[List["SchemaOrBool"]] = None
    # TODO: uncomment and remove below when deprecating Pydantic v1
    # It generales a list of schemas for tuples, before prefixItems was available
    # items: Optional["SchemaOrBool"] = None
    items: Optional[Union["SchemaOrBool", List["SchemaOrBool"]]] = None
    contains: Optional["SchemaOrBool"] = None
    properties: Optional[Dict[str, "SchemaOrBool"]] = None
    patternProperties: Optional[Dict[str, "SchemaOrBool"]] = None
    additionalProperties: Optional["SchemaOrBool"] = None
    propertyNames: Optional["SchemaOrBool"] = None
    unevaluatedItems: Optional["SchemaOrBool"] = None
    unevaluatedProperties: Optional["SchemaOrBool"] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-structural
    # A Vocabulary for Structural Validation
    type: Optional[str] = None
    enum: Optional[List[Any]] = None
    const: Optional[Any] = None
    multipleOf: Optional[float] = Field_V1(default=None, gt=0)
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    maxLength: Optional[int] = Field_V1(default=None, ge=0)
    minLength: Optional[int] = Field_V1(default=None, ge=0)
    pattern: Optional[str] = None
    maxItems: Optional[int] = Field_V1(default=None, ge=0)
    minItems: Optional[int] = Field_V1(default=None, ge=0)
    uniqueItems: Optional[bool] = None
    maxContains: Optional[int] = Field_V1(default=None, ge=0)
    minContains: Optional[int] = Field_V1(default=None, ge=0)
    maxProperties: Optional[int] = Field_V1(default=None, ge=0)
    minProperties: Optional[int] = Field_V1(default=None, ge=0)
    required: Optional[List[str]] = None
    dependentRequired: Optional[Dict[str, Set[str]]] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-vocabularies-for-semantic-c
    # Vocabularies for Semantic Content With "format"
    format: Optional[str] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-the-conten
    # A Vocabulary for the Contents of String-Encoded Data
    contentEncoding: Optional[str] = None
    contentMediaType: Optional[str] = None
    contentSchema: Optional["SchemaOrBool"] = None
    # Ref: JSON Schema Validation 2020-12: https://json-schema.org/draft/2020-12/json-schema-validation.html#name-a-vocabulary-for-basic-meta
    # A Vocabulary for Basic Meta-Data Annotations
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    deprecated: Optional[bool] = None
    readOnly: Optional[bool] = None
    writeOnly: Optional[bool] = None
    examples: Optional[List[Any]] = None
    # Ref: OpenAPI 3.1.0: https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#schema-object
    # Schema Object
    discriminator: Optional[Discriminator] = None
    xml: Optional[XML] = None
    externalDocs: Optional[ExternalDocumentation] = None
    example: Annotated[
        Optional[Any],
        typing_deprecated(
            "Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, "
            "although still supported. Use examples instead."
        ),
    ] = None


# Ref: https://json-schema.org/draft/2020-12/json-schema-core.html#name-json-schema-documents
# A JSON Schema MUST be an object or a boolean.
SchemaOrBool = Union[Schema, Schema_V1, bool]


class Example(TypedDict, total=False):
    summary: Optional[str]
    description: Optional[str]
    value: Optional[Any]
    externalValue: Optional[AnyUrl]

    if PYDANTIC_V2:  # type: ignore [misc]
        __pydantic_config__ = {"extra": "allow"}

    else:

        class Config:
            extra = "allow"

class Example_V1(TypedDict, total=False):
    summary: Optional[str]
    description: Optional[str]
    value: Optional[Any]
    externalValue: Optional[AnyUrl_V1]

    class Config:
        extra = "allow"


class ParameterInType(Enum):
    query = "query"
    header = "header"
    path = "path"
    cookie = "cookie"


class Encoding(BaseModelWithConfig):
    contentType: Optional[str] = None
    headers: Optional[Dict[str, Union["Header", Reference]]] = None
    style: Optional[str] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = None

class Encoding_V1(BaseModelWithConfig_V1):
    contentType: Optional[str] = None
    headers: Optional[Dict[str, Union["Header_V1", Reference_V1]]] = None
    style: Optional[str] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = None


class MediaType(BaseModelWithConfig):
    schema_: Optional[Union[Schema, Reference]] = Field(default=None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None
    encoding: Optional[Dict[str, Encoding]] = None


class MediaType_V1(BaseModelWithConfig_V1):
    schema_: Optional[Union[Schema_V1, Reference_V1]] = Field_V1(default=None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example_V1, Reference_V1]]] = None
    encoding: Optional[Dict[str, Encoding_V1]] = None


class ParameterBase(BaseModelWithConfig):
    description: Optional[str] = None
    required: Optional[bool] = None
    deprecated: Optional[bool] = None
    # Serialization rules for simple scenarios
    style: Optional[str] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = None
    schema_: Optional[Union[Schema, Reference]] = Field(default=None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None
    # Serialization rules for more complex scenarios
    content: Optional[Dict[str, MediaType]] = None


class ParameterBase_V1(BaseModelWithConfig_V1):
    description: Optional[str] = None
    required: Optional[bool] = None
    deprecated: Optional[bool] = None
    # Serialization rules for simple scenarios
    style: Optional[str] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = None
    schema_: Optional[Union[Schema_V1, Reference_V1]] = Field_V1(default=None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example_V1, Reference_V1]]] = None
    # Serialization rules for more complex scenarios
    content: Optional[Dict[str, MediaType_V1]] = None


class Parameter(ParameterBase):
    name: str
    in_: ParameterInType = Field(alias="in")


class Parameter_V1(ParameterBase_V1):
    name: str
    in_: ParameterInType = Field_V1(alias="in")


class Header(ParameterBase):
    pass

class Header_V1(ParameterBase_V1):
    pass


class RequestBody(BaseModelWithConfig):
    description: Optional[str] = None
    content: Dict[str, MediaType]
    required: Optional[bool] = None


class RequestBody_V1(BaseModelWithConfig_V1):
    description: Optional[str] = None
    content: Dict[str, MediaType_V1]
    required: Optional[bool] = None


class Link(BaseModelWithConfig):
    operationRef: Optional[str] = None
    operationId: Optional[str] = None
    parameters: Optional[Dict[str, Union[Any, str]]] = None
    requestBody: Optional[Union[Any, str]] = None
    description: Optional[str] = None
    server: Optional[Server] = None


class Link_V1(BaseModelWithConfig_V1):
    operationRef: Optional[str] = None
    operationId: Optional[str] = None
    parameters: Optional[Dict[str, Union[Any, str]]] = None
    requestBody: Optional[Union[Any, str]] = None
    description: Optional[str] = None
    server: Optional[Server_V1] = None


class Response(BaseModelWithConfig):
    description: str
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    content: Optional[Dict[str, MediaType]] = None
    links: Optional[Dict[str, Union[Link, Reference]]] = None


class Response_V1(BaseModelWithConfig_V1):
    description: str
    headers: Optional[Dict[str, Union[Header_V1, Reference_V1]]] = None
    content: Optional[Dict[str, MediaType_V1]] = None
    links: Optional[Dict[str, Union[Link_V1, Reference_V1]]] = None


class Operation(BaseModelWithConfig):
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None
    operationId: Optional[str] = None
    parameters: Optional[List[Union[Parameter, Reference]]] = None
    requestBody: Optional[Union[RequestBody, Reference]] = None
    # Using Any for Specification Extensions
    responses: Optional[Dict[str, Union[Response, Any]]] = None
    callbacks: Optional[Dict[str, Union[Dict[str, "PathItem"], Reference]]] = None
    deprecated: Optional[bool] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    servers: Optional[List[Server]] = None


class Operation_V1(BaseModelWithConfig_V1):
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation_V1] = None
    operationId: Optional[str] = None
    parameters: Optional[List[Union[Parameter_V1, Reference_V1]]] = None
    requestBody: Optional[Union[RequestBody_V1, Reference_V1]] = None
    # Using Any for Specification Extensions
    responses: Optional[Dict[str, Union[Response_V1, Any]]] = None
    callbacks: Optional[Dict[str, Union[Dict[str, "PathItem_V1"], Reference_V1]]] = None
    deprecated: Optional[bool] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    servers: Optional[List[Server_V1]] = None


class PathItem(BaseModelWithConfig):
    ref: Optional[str] = Field(default=None, alias="$ref")
    summary: Optional[str] = None
    description: Optional[str] = None
    get: Optional[Operation] = None
    put: Optional[Operation] = None
    post: Optional[Operation] = None
    delete: Optional[Operation] = None
    options: Optional[Operation] = None
    head: Optional[Operation] = None
    patch: Optional[Operation] = None
    trace: Optional[Operation] = None
    servers: Optional[List[Server]] = None
    parameters: Optional[List[Union[Parameter, Reference]]] = None


class PathItem_V1(BaseModelWithConfig_V1):
    ref: Optional[str] = Field_V1(default=None, alias="$ref")
    summary: Optional[str] = None
    description: Optional[str] = None
    get: Optional[Operation_V1] = None
    put: Optional[Operation_V1] = None
    post: Optional[Operation_V1] = None
    delete: Optional[Operation_V1] = None
    options: Optional[Operation_V1] = None
    head: Optional[Operation_V1] = None
    patch: Optional[Operation_V1] = None
    trace: Optional[Operation_V1] = None
    servers: Optional[List[Server_V1]] = None
    parameters: Optional[List[Union[Parameter_V1, Reference_V1]]] = None


class SecuritySchemeType(Enum):
    apiKey = "apiKey"
    http = "http"
    oauth2 = "oauth2"
    openIdConnect = "openIdConnect"


class SecurityBase(BaseModelWithConfig):
    type_: SecuritySchemeType = Field(alias="type")
    description: Optional[str] = None


class SecurityBase_V1(BaseModelWithConfig_V1):
    type_: SecuritySchemeType = Field_V1(alias="type")
    description: Optional[str] = None



class APIKeyIn(Enum):
    query = "query"
    header = "header"
    cookie = "cookie"


class APIKey(SecurityBase):
    type_: SecuritySchemeType = Field(default=SecuritySchemeType.apiKey, alias="type")
    in_: APIKeyIn = Field(alias="in")
    name: str


class APIKey_V1(SecurityBase_V1):
    type_: SecuritySchemeType = Field_V1(default=SecuritySchemeType.apiKey, alias="type")
    in_: APIKeyIn = Field_V1(alias="in")
    name: str


class HTTPBase(SecurityBase):
    type_: SecuritySchemeType = Field(default=SecuritySchemeType.http, alias="type")
    scheme: str


class HTTPBase_V1(SecurityBase_V1):
    type_: SecuritySchemeType = Field_V1(default=SecuritySchemeType.http, alias="type")
    scheme: str


class HTTPBearer(HTTPBase):
    scheme: Literal["bearer"] = "bearer"
    bearerFormat: Optional[str] = None


class HTTPBearer_V1(HTTPBase_V1):
    scheme: Literal["bearer"] = "bearer"
    bearerFormat: Optional[str] = None


class OAuthFlow(BaseModelWithConfig):
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str] = {}


class OAuthFlow_V1(BaseModelWithConfig_V1):
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str] = {}


class OAuthFlowImplicit(OAuthFlow):
    authorizationUrl: str


class OAuthFlowImplicit_V1(OAuthFlow_V1):
    authorizationUrl: str


class OAuthFlowPassword(OAuthFlow):
    tokenUrl: str


class OAuthFlowPassword_V1(OAuthFlow_V1):
    tokenUrl: str


class OAuthFlowClientCredentials(OAuthFlow):
    tokenUrl: str


class OAuthFlowClientCredentials_V1(OAuthFlow_V1):
    tokenUrl: str


class OAuthFlowAuthorizationCode(OAuthFlow):
    authorizationUrl: str
    tokenUrl: str


class OAuthFlowAuthorizationCode_V1(OAuthFlow_V1):
    authorizationUrl: str
    tokenUrl: str


class OAuthFlows(BaseModelWithConfig):
    implicit: Optional[OAuthFlowImplicit] = None
    password: Optional[OAuthFlowPassword] = None
    clientCredentials: Optional[OAuthFlowClientCredentials] = None
    authorizationCode: Optional[OAuthFlowAuthorizationCode] = None


class OAuthFlows_V1(BaseModelWithConfig_V1):
    implicit: Optional[OAuthFlowImplicit_V1] = None
    password: Optional[OAuthFlowPassword_V1] = None
    clientCredentials: Optional[OAuthFlowClientCredentials_V1] = None
    authorizationCode: Optional[OAuthFlowAuthorizationCode_V1] = None


class OAuth2(SecurityBase):
    type_: SecuritySchemeType = Field(default=SecuritySchemeType.oauth2, alias="type")
    flows: OAuthFlows


class OAuth2_V1(SecurityBase_V1):
    type_: SecuritySchemeType = Field_V1(default=SecuritySchemeType.oauth2, alias="type")
    flows: OAuthFlows_V1


class OpenIdConnect(SecurityBase):
    type_: SecuritySchemeType = Field(
        default=SecuritySchemeType.openIdConnect, alias="type"
    )
    openIdConnectUrl: str


class OpenIdConnect_V1(SecurityBase_V1):
    type_: SecuritySchemeType = Field_V1(
        default=SecuritySchemeType.openIdConnect, alias="type"
    )
    openIdConnectUrl: str


SecurityScheme = Union[APIKey, HTTPBase, OAuth2, OpenIdConnect, HTTPBearer, APIKey_V1, HTTPBase_V1, OAuth2_V1, OpenIdConnect_V1, HTTPBearer_V1]


class Components(BaseModelWithConfig):
    schemas: Optional[Dict[str, Union[Schema, Reference]]] = None
    responses: Optional[Dict[str, Union[Response, Reference]]] = None
    parameters: Optional[Dict[str, Union[Parameter, Reference]]] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None
    requestBodies: Optional[Dict[str, Union[RequestBody, Reference]]] = None
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    securitySchemes: Optional[Dict[str, Union[SecurityScheme, Reference]]] = None
    links: Optional[Dict[str, Union[Link, Reference]]] = None
    # Using Any for Specification Extensions
    callbacks: Optional[Dict[str, Union[Dict[str, PathItem], Reference, Any]]] = None
    pathItems: Optional[Dict[str, Union[PathItem, Reference]]] = None


class Components_V1(BaseModelWithConfig_V1):
    schemas: Optional[Dict[str, Union[Schema_V1, Reference_V1]]] = None
    responses: Optional[Dict[str, Union[Response_V1, Reference_V1]]] = None
    parameters: Optional[Dict[str, Union[Parameter_V1, Reference_V1]]] = None
    examples: Optional[Dict[str, Union[Example_V1, Reference_V1]]] = None
    requestBodies: Optional[Dict[str, Union[RequestBody_V1, Reference_V1]]] = None
    headers: Optional[Dict[str, Union[Header_V1, Reference_V1]]] = None
    securitySchemes: Optional[Dict[str, Union[SecurityScheme, Reference_V1]]] = None
    links: Optional[Dict[str, Union[Link_V1, Reference_V1]]] = None
    # Using Any for Specification Extensions
    callbacks: Optional[Dict[str, Union[Dict[str, PathItem_V1], Reference_V1, Any]]] = None
    pathItems: Optional[Dict[str, Union[PathItem_V1, Reference_V1]]] = None


class Tag(BaseModelWithConfig):
    name: str
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None


class Tag_V1(BaseModelWithConfig_V1):
    name: str
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation_V1] = None

class OpenAPI_V2(BaseModelWithConfig):
    openapi: str
    info: Info
    jsonSchemaDialect: Optional[str] = None
    servers: Optional[List[Server]] = None
    # Using Any for Specification Extensions
    paths: Optional[Dict[str, Union[PathItem, Any]]] = None
    webhooks: Optional[Dict[str, Union[PathItem, Reference]]] = None
    components: Optional[Components] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    tags: Optional[List[Tag]] = None
    externalDocs: Optional[ExternalDocumentation] = None


class OpenAPI_V1(BaseModelWithConfig_V1):
    openapi: str
    info: Info_V1
    jsonSchemaDialect: Optional[str] = None
    servers: Optional[List[Server_V1]] = None
    # Using Any for Specification Extensions
    paths: Optional[Dict[str, Union[PathItem_V1, Any]]] = None
    webhooks: Optional[Dict[str, Union[PathItem_V1, Reference_V1]]] = None
    components: Optional[Components_V1] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    tags: Optional[List[Tag_V1]] = None
    externalDocs: Optional[ExternalDocumentation_V1] = None


class OpenAPI(NamedTuple):
    openapi: str
    info: Info
    jsonSchemaDialect: Optional[str] = None
    servers: Optional[List[Union[Server, Server_V1]]] = None
    # Using Any for Specification Extensions
    paths: Optional[Dict[str, Union[PathItem, PathItem_V1, Any]]] = None
    webhooks: Optional[Dict[str, Union[PathItem, PathItem_V1, Reference, Reference_V1]]] = None
    components: Optional[Union[Components, Components_V1]] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    tags: Optional[List[Union[Tag, Tag_V1]]] = None
    externalDocs: Optional[Union[ExternalDocumentation, ExternalDocumentation_V1]] = None


_model_rebuild(Schema)
_model_rebuild(Schema_V1)
_model_rebuild(Operation)
_model_rebuild(Operation_V1)
_model_rebuild(Encoding)
_model_rebuild(Encoding_V1)
