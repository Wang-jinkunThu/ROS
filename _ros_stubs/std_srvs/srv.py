"""std_srvs.srv stub for Windows editing only."""


class SetBoolRequest:
    data: bool = False


class SetBoolResponse:
    success: bool = False
    message: str = ""


class SetBool:
    Request = SetBoolRequest
    Response = SetBoolResponse
