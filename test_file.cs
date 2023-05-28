

using System;
using System.Net;
using System.Net.Http;
using System.Web.Http;

public class DerpController : ApiController
{
    [HttpGet]
    [Route("/derp")]
    public HttpResponseMessage GetDerp()
    {
        var response = new HttpResponseMessage(HttpStatusCode.OK);
        response.Content = new StringContent("{\"message\": \"hello nerd\"}", System.Text.Encoding.UTF8, "application/json");
        return response;
    }
}